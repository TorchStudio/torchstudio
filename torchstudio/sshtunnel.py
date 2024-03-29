import time
import sys
import os
import io

# Port forwarding from https://github.com/skyleronken/sshrat/blob/master/tunnels.py
# improved with:
# dynamic local port allocation feedback for reverse tunnel with a null local port
# blocking connections to avoid connection lost with poor cloud servers
# more explicit error messages
import threading
import socket
import selectors
import time
import socketserver
import paramiko
import hashlib

class Tunnel():

    def __init__(self, ssh_session, tun_type, lhost, lport, dhost, dport):
        self.tun_type = tun_type
        self.lhost = lhost
        self.lport = lport
        self.dhost = dhost
        self.dport = dport

        # create tunnel here
        if self.tun_type == ForwardTunnel:
            self.tunnel = ForwardTunnel(ssh_session, self.lhost, self.lport, self.dhost, self.dport)
        elif self.tun_type == ReverseTunnel:
            self.tunnel = ReverseTunnel(ssh_session, self.lhost, self.lport, self.dhost, self.dport)
            self.lport = self.tunnel.lport #in case of dynamic allocation (lport=0)

    def to_str(self):
        if self.tun_type == ForwardTunnel:
            return f"{self.lhost}:{self.lport} --> {self.dhost}:{self.dport}"
        else:
            return f"{self.dhost}:{self.dport} <-- {self.lhost}:{self.lport}"

    def stop(self):
        self.tunnel.stop()

class ReverseTunnel():

    def __init__(self, ssh_session, lhost, lport, dhost, dport):
        self.session = ssh_session
        self.lhost = lhost
        self.lport = lport
        self.dhost = dhost
        self.dport = dport

        self.transport = ssh_session.get_transport()

        self.reverse_forward_tunnel(lhost, lport, dhost, dport, self.transport)
        self.handlers = []

    def stop(self):
        self.transport.cancel_port_forward(self.lhost, self.lport)
        for thr in self.handlers:
            thr.stop()

    def handler(self, rev_socket, origin, laddress):
        rev_handler = ReverseTunnelHandler(rev_socket, self.dhost, self.dport, self.lhost, self.lport)
        rev_handler.daemon=True
        rev_handler.start()
        self.handlers.append(rev_handler)

    def reverse_forward_tunnel(self, lhost, lport, dhost, dport, transport):
        try:
            self.lport=transport.request_port_forward(lhost, lport, handler=self.handler)
        except Exception as e:
            raise e

class ReverseTunnelHandler(threading.Thread):

    def __init__(self, rev_socket, dhost, dport, lhost, lport):

        threading.Thread.__init__(self)

        self.rev_socket = rev_socket
        self.dhost = dhost
        self.dport = dport
        self.lhost = lhost
        self.lport = lport

        self.dst_socket = socket.socket()
        try:
            self.dst_socket.connect((self.dhost, self.dport))
        except Exception as e:
            raise e

        self.keepalive = True

    def _read_from_rev(self, dst, rev):
        self._transfer_data(src_socket=rev,dest_socket=dst)

    def _read_from_dest(self, dst, rev):
        self._transfer_data(src_socket=dst,dest_socket=rev)

    def _transfer_data(self,src_socket,dest_socket):
        dest_socket.setblocking(True)
        data = src_socket.recv(1024)

        if len(data):
            try:
                dest_socket.send(data)
            except Exception as e:
                print(f"ssh error: {type(e).__name__}", file=sys.stderr)

    def stop(self):
        self.rev_socket.shutdown(2)
        self.dst_socket.shutdown(2)
        self.rev_socket.close()
        self.dst_socket.close()
        self.keepalive = False

    def run(self):
        selector = selectors.DefaultSelector()

        selector.register(fileobj=self.rev_socket,events=selectors.EVENT_READ,data=self._read_from_rev)
        selector.register(fileobj=self.dst_socket,events=selectors.EVENT_READ,data=self._read_from_dest)

        while self.keepalive:
            events = selector.select(5)
            if len(events) > 0:
                for key, _ in events:
                    callback = key.data
                    try:
                        callback(dst=self.dst_socket,rev=self.rev_socket)
                    except Exception as e:
                        print(f"ssh error: {type(e).__name__}", file=sys.stderr)
                time.sleep(0)



# credits to paramiko-tunnel
class ForwardTunnel(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, ssh_session, lhost, lport, dhost, dport):
        self.session = ssh_session
        self.lhost = lhost
        self.lport = lport
        self.dhost = dhost
        self.dport = dport

        super().__init__(
            server_address=(lhost, lport),
            RequestHandlerClass=ForwardTunnelHandler,
            bind_and_activate=True,
        )

        self.baddr, self.bport = self.server_address
        self.thread = threading.Thread(
            target=self.serve_forever,
            daemon=True,
        )

        self.start()

    def start(self):
        self.thread.start()

    def stop(self):
        self.shutdown()
        self.server_close()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

class ForwardTunnelHandler(socketserver.BaseRequestHandler):
    sz_buf = 1024

    def __init__(self, request, cli_addr, server):
        self.selector = selectors.DefaultSelector()
        self.channel = None
        super().__init__(request, cli_addr, server)

    def _read_from_client(self, sock, mask):
        self._transfer_data(src_socket=sock, dest_socket=self.channel)

    def _read_from_channel(self, sock, mask):
        self._transfer_data(src_socket=sock,dest_socket=self.request)

    def _transfer_data(self,src_socket,dest_socket):
        src_socket.setblocking(True)
        data = src_socket.recv(self.sz_buf)

        if len(data):
            try:
                dest_socket.send(data)
            except BrokenPipeError:
                self.finish()

    def handle(self):
        peer_name = self.request.getpeername()
        try:
            self.channel = self.server.session.get_transport().open_channel(
                kind='direct-tcpip',
                dest_addr=(self.server.dhost,self.server.dport,),
                src_addr=peer_name,
            )
        except Exception as error:
            msg = f'Connection failed to {self.server.dhost}:{self.server.dport}'
            raise Exception(msg)

        else:
            self.selector.register(fileobj=self.channel,events=selectors.EVENT_READ,data=self._read_from_channel)
            self.selector.register(fileobj=self.request,events=selectors.EVENT_READ,data=self._read_from_client)

            if self.channel is None:
                self.finish()
                raise Exception(f'SSH Server rejected request to {self.server.dhost}:{self.server.dport}')

            while True:
                events = self.selector.select()
                for key, mask in events:
                    callback = key.data
                    callback(sock=key.fileobj,mask=mask)
                    if self.server._BaseServer__is_shut_down.is_set():
                        self.finish()
                time.sleep(0)

    def finish(self):
        if self.channel is not None:
            self.channel.shutdown(how=2)
            self.channel.close()
        self.request.shutdown(2)
        self.request.close()

###



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sshaddress", help="server address", type=str, default=None)
    parser.add_argument("--sshport", help="ssh server port", type=int, default=22)
    parser.add_argument("--username", help="ssh server username", type=str, default=None)
    parser.add_argument("--password", help="ssh server password", type=str, default=None)
    parser.add_argument("--keyfile", help="ssh server key file", type=str, default=None)
    parser.add_argument('--clean', help="cleaning level (0: cache, 1: environment, 2: all)", type=int, default=None)
    parser.add_argument("--command", help="command to execute or run python scripts", type=str, default=None)
    parser.add_argument("--script", help="python script to be launched on the server", type=str, default=None)
    parser.add_argument("--address", help="address to which the script must connect", type=str, default=None)
    parser.add_argument("--port", help="port to which the script must connect", type=int, default=None)
    args, other_args = parser.parse_known_args()

    sshclient = paramiko.SSHClient()
    sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("Connecting to remote server...", file=sys.stderr)
    try:
        sshclient.connect(hostname=args.sshaddress, port=args.sshport, username=args.username, password=args.password, pkey=paramiko.RSAKey.from_private_key_file(args.keyfile) if args.keyfile else None, timeout=10)
    except:
        print("Error: could not connect to remote server", file=sys.stderr)
        exit(1)

    if args.clean is not None:
        if args.clean==0:
            print("Cleaning TorchStudio cache...", file=sys.stderr)
            stdin, stdout, stderr = sshclient.exec_command('rm -r -f TorchStudio/cache')
            exit_status = stdout.channel.recv_exit_status()
            stdin, stdout, stderr = sshclient.exec_command('rmdir /s /q TorchStudio\\cache')
            exit_status = stdout.channel.recv_exit_status()
        if args.clean==1:
            print("Deleting TorchStudio environment...", file=sys.stderr)
            stdin, stdout, stderr = sshclient.exec_command('rm -r -f TorchStudio/python')
            exit_status = stdout.channel.recv_exit_status()
            stdin, stdout, stderr = sshclient.exec_command('rmdir /s /q TorchStudio\\python')
            exit_status = stdout.channel.recv_exit_status()
        if args.clean==2:
            print("Deleting all TorchStudio files...", file=sys.stderr)
            stdin, stdout, stderr = sshclient.exec_command('rm -r -f TorchStudio')
            exit_status = stdout.channel.recv_exit_status()
            stdin, stdout, stderr = sshclient.exec_command('rmdir /s /q TorchStudio')
            exit_status = stdout.channel.recv_exit_status()
        sshclient.close()
        print("Cleaning complete")
        exit(0)

    #copy root scripts to the remote server if necessary
    print("Updating remote scripts...", file=sys.stderr)
    local_scripts_hash = hashlib.md5()
    for entry in os.scandir('torchstudio'):
        if entry.is_file():
            with open('torchstudio/'+entry.name, 'rb') as f:
                local_scripts_hash.update(f.read())
    local_scripts_hash = local_scripts_hash.digest()
    sftp = paramiko.SFTPClient.from_transport(sshclient.get_transport())
    remote_scripts_hash=io.BytesIO()
    try:
        sftp.getfo('TorchStudio/torchstudio/.md5', remote_scripts_hash)
    except:
        pass
    if remote_scripts_hash.getvalue()!=local_scripts_hash:
        try:
            sftp.mkdir('TorchStudio')
        except:
            pass
        try:
            sftp.mkdir('TorchStudio/torchstudio')
        except:
            pass
        try:
            for entry in os.scandir('torchstudio'):
                if entry.is_file():
                    sftp.put('torchstudio/'+entry.name, 'TorchStudio/torchstudio/'+entry.name)
                    if entry.name.endswith('.cmd'):
                        sftp.chmod('TorchStudio/torchstudio/'+entry.name, 0o0777)
        except:
            print("Error: could not update remote scripts", file=sys.stderr)
            exit(1)
        new_scripts_hash=io.BytesIO(local_scripts_hash)
        sftp.putfo(new_scripts_hash, 'TorchStudio/torchstudio/.md5')
    sftp.close()

    if args.address:
        other_args=["--address", args.address]+other_args

    if args.port:
        #setup remote port forwarding
        print("Forwarding ports...", file=sys.stderr)
        reverse_tunnel = Tunnel(sshclient, ReverseTunnel, 'localhost', 0, args.address if args.address else 'localhost', args.port) #remote address, remote port, local address, local port
        other_args=["--port", str(reverse_tunnel.lport)]+other_args

    if args.command:
        if args.script:
            print("Launching remote script...", file=sys.stderr)
            if '\\python' in args.command: #python on Windows, add path variables
                python_root=args.command[:args.command.rindex('\\python')]
                if '&&' in python_root:
                    python_root=python_root[python_root.rindex('&&')+2:]
                args.command='set PATH=%PATH%;'+python_root+';'+python_root+'\\Library\\mingw-w64\\bin;'+python_root+'\\Library\\bin;'+python_root+'\\bin&&'+args.command
            stdin, stdout, stderr = sshclient.exec_command("cd TorchStudio&&"+args.command+" -u -X utf8 -m "+' '.join([args.script]+other_args))
        else:
            print("Executing remote command...", file=sys.stderr)
            stdin, stdout, stderr = sshclient.exec_command("cd TorchStudio&&"+' '.join([args.command]+other_args))

        while not stdout.channel.exit_status_ready():
            time.sleep(.01) #lower CPU usage
            if stdout.channel.recv_stderr_ready():
                sys.stderr.buffer.write(stdout.channel.recv_stderr(8192))
                time.sleep(.01) #for stdout/stderr sync
            if stdout.channel.recv_ready():
                sys.stdout.buffer.write(stdout.channel.recv(8192))
                time.sleep(.01) #for stdout/stderr sync
    else:
        if args.script:
            print("Error: no python environment set.", file=sys.stderr)
        else:
            print("Error: no command set.", file=sys.stderr)

    sshclient.close()


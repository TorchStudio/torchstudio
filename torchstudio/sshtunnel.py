import time
import sys
import os
import io

# Port forwarding from https://github.com/skyleronken/sshrat/blob/master/tunnels.py
# improved with dynamic local port allocation feedback for reverse tunnel with a null local port
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
        rev_handler.setDaemon(True)
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
    parser.add_argument("--command", help="command to run python scripts", type=str, default="python")
    parser.add_argument("--script", help="script to be launched on the server", type=str, default=None)
    parser.add_argument("--address", help="address to which the script must connect", type=str, default=None)
    parser.add_argument("--port", help="port to which the script must connect", type=int, default=None)
    args, other_args = parser.parse_known_args()

    sshclient = paramiko.SSHClient()
    sshclient.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    print("Connecting to remote server...", file=sys.stderr)
    try:
        sshclient.connect(hostname=args.sshaddress, port=args.sshport, username=args.username, password=args.password, pkey=paramiko.RSAKey.from_private_key_file(args.keyfile) if args.keyfile else None, timeout=5)
    except:
        print("Error: could not connect to remote server", file=sys.stderr)
        exit()

    #copy root scripts to the remote server if necessary
    print("Updating remote scripts...", file=sys.stderr)
    local_scripts_hash = hashlib.md5()
    for filename in os.listdir('torchstudio'):
        if filename.endswith('.py'):
            with open('torchstudio/'+filename, 'rb') as f:
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
        for filename in os.listdir('torchstudio'):
            if filename.endswith('.py'):
                sftp.put('torchstudio/'+filename, 'TorchStudio/torchstudio/'+filename)
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

    if args.script:
        print("Launching remote script...", file=sys.stderr)
        stdin, stdout, stderr = sshclient.exec_command("cd TorchStudio\n"+args.command+" -u -X utf8 -m "+' '.join([args.script]+other_args))
        while True:
            time.sleep(.1)
            if stdout.channel.recv_ready():
                sys.stdout.write(str(stdout.channel.recv(1024),'utf-8'))
            if stdout.channel.recv_stderr_ready():
                sys.stderr.write(str(stdout.channel.recv_stderr(1024),'utf-8'))
            if stdout.channel.exit_status_ready():
                break

    sshclient.close()


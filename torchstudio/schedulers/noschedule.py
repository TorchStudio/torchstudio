class NoSchedule():
    """No Schedule
    """
    def __init__(self,
                 optimizer,
                 last_epoch=-1):
        self.last_epoch=0 if last_epoch<0 else last_epoch

    def step(self):
        self.last_epoch+=1



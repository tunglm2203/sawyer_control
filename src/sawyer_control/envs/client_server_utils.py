import zmq
import json
import time

SERVER_ENDPOINT = 'tcp://127.0.0.1:5555'
DEBUG = True


# TUNG: Run on python 2
class ServerProcess:
    """
        Support to listen all REQ from client, where the REQ includes `function` and `its arguments`
    """

    def __init__(self, cmd_list=[], func_list=[]):
        self.cmd_dict = dict(zip(cmd_list, func_list))
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind(SERVER_ENDPOINT)

    def listening(self):
        while True:
            msg = self.socket.recv()
            try:
                _tmp = json.loads(msg, encoding='utf-8')
            except Exception:
                _tmp = None
            if isinstance(_tmp, dict):
                msg = _tmp
                assert 'func' in msg.keys() and 'args' in msg.keys(), \
                    '[ERROR] Wrong format of message'
                if msg['func'] in self.cmd_dict.keys():
                    # Do something with msg
                    if DEBUG:
                        print('[DEBUG] Server: Received message: ', msg)
                    self.cmd_dict[msg['func']](**msg['args'])
                    self.socket.send('recieved')
                else:
                    print('[WARNING] Not have function {}() in {}'.format(msg['func'],
                                                                          self.cmd_dict.keys()))
                    self.socket.send('recieved')
            elif msg == 'close':
                if DEBUG:
                    print('[DEBUG] Server: Close listener.')
                self.socket.send('closed')
                self.socket.close()
                break
            else:
                print('[WARNING] Not support message: ', msg)
                self.socket.send('recieved')


# TUNG: Run on python 3
class ClientProcess:
    """
        Support to request Server (Listener) execute some functions. Structure of message is dict with
        two keys:
            `func`: value is string
            `args`: dict of arguments (depend on function's argument type)
                e.g.: args = {'x': 0.5, 'y': 0.0, 'z': 0.87}
    """

    def __init__(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(SERVER_ENDPOINT)

    def sending(self, msg='hello world', sleep_before=0.0, sleep_after=0.0):
        if isinstance(msg, dict):
            assert 'func' in msg.keys() and 'args' in msg.keys(), '[ERROR] Wrong format of message'
            time.sleep(sleep_before)
            self.socket.send_string(json.dumps(msg), encoding='utf-8')
            time.sleep(sleep_after)
        elif isinstance(msg, str):
            time.sleep(sleep_before)
            self.socket.send_string(msg, encoding='utf-8')
            time.sleep(sleep_after)
        else:
            print('[ERROR] Not support type of message.')
            exit()
        msg = self.socket.recv()
        if DEBUG:
            print('[DEBUG] Client: Message from Server: ', msg)

    def close(self):
        self.sending(msg='close')

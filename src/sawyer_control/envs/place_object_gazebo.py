#!/usr/bin/env python
from sawyer_control.envs.client_server_utils import ServerProcess
from subprocess import call

bash_script_path = '/home/tung/ros_ws/src/sawyer_control/src/sawyer_control/envs/set_object_loc.sh'


# Support set object in XYZ-plane
def set_object_los(x, y, z):
    file_args = [bash_script_path,
                 str(x),
                 str(y),
                 str(z)]
    try:
        call(file_args)
        return 0
    except Exception:
        print('Unknow error when calling bash script.')
        return 1


def main():
    cmd_list = ['set_object_los']
    func_list = [set_object_los]
    listener = ServerProcess(cmd_list=cmd_list, func_list=func_list)
    listener.listening()


if __name__ == '__main__':
    main()



import time
import subprocess
import os


class SofaRecorder(object):

    def __init__(self):
        self.x_server = None
        self.video_dir = '/builds/sofa/screenshots'
        self.start_x_server()

    def start_x_server(self):
        if self.x_server is None and not self.x_server_exists():
            self.x_server = subprocess.Popen(
                ["X", "-config", "/dummy.conf"]
            )
            time.sleep(3)  # wait for process to get online

        # pyautogui cannot be imported until there is a display to connect to process launches
        import pyautogui
        self.pyautogui = pyautogui

    def x_server_exists(self):
        script = 'if ! timeout 1s xset q 1>/dev/null; then exit 1; fi'
        try:
            subprocess.run(script, shell=True, check=True)
            return True
        except subprocess.CalledProcessError as e:
            print(e)
            return False

    def start_recording(self):
        time.sleep(2)
        self.pyautogui.write('v')

    def stop_recording(self):
        self.pyautogui.write('v')
        time.sleep(2)

    def stop_x_server(self):
        if self.x_server is not None:
            self.x_server.kill()
            # subprocess.Popen(["sudo", "kill", str(self.x_server.pid)])
            time.sleep(1)
            self.x_server = None

    def delete_recordings(self):
        for file in os.listdir(self.video_dir):
            os.remove(os.path.join(self.video_dir, file))

    def merge_recordings(self, outfile):
        files = sorted(list(os.listdir(self.video_dir)))
        files = [os.path.join(self.video_dir, f) for f in files]

        vid_list = '/tmp/video_list.txt'
        with open(vid_list, 'w') as f:
            for file in files:
                f.write(f"file '{file}' + \n")
        cmd = f'ffmpeg -y -f concat -safe 0 -i {vid_list} -c copy {outfile}'
        subprocess.run(cmd, shell=True)
        os.remove(vid_list)

    def __del__(self):
        if self.x_server is not None:
            self.stop_x_server()


# example of usage
if __name__ == "__main__":
    recorder = SofaRecorder()
    # if the x_server is not started before the sofa gui launches
    # the sofa process will seg fault
    # without the -g batch option this launches a sofa gui
    sofa_process = subprocess.Popen(["runSofa", "--start"])
    recorder.start_recording()
    print('recording started')
    time.sleep(10) #note this records for 10 wall clock seconds not seconds in the simulator
    recorder.stop_recording()
    print('recording stoppped')
    time.sleep(3)
    sofa_process.terminate()
    print("sofa terminated")
    time.sleep(3)
    recorder.stop_x_server()
    print('x server terminated')

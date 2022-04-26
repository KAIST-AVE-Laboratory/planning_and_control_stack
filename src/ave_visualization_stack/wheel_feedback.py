import evdev
from evdev import ecodes, InputDevice, ff
from carla_msgs.msg import CarlaEgoVehicleControl
import numpy as np
import rospy
import threading
import time

class WheelFeedback():

    def __init__(self,
        role_name: str,
        wheel_name: str,
        pid_coef: list) -> None:
        joystick_list = evdev.list_devices()
        if len(joystick_list)==0:
            raise ValueError('No wheel connected!')
        elif len(joystick_list)>1:
            raise ValueError('Connect only one wheel!')
        
        wheel = InputDevice(evdev.list_devices()[0])
        wheel.read()
        print('wheel connected: ', wheel)

        self.wheel = wheel
        self.pid_coef = np.array([pid_coef], dtype=np.float32)
        self.pid_err = np.array([0, 0, 0], dtype=np.float32)
        self.envelope = ff.Envelope(0, 0, 0, 0)
        self.center = 65535//2
        self.duration_ms = 100

        self.cmd_subscriber = rospy.Subscriber(
            f'/carla/{role_name}/vehicle_control_cmd_manual',
            CarlaEgoVehicleControl,
            self._cmd_callback,
            queue_size=1
        )
        self._lock=threading.Lock()

    def _cmd_callback(self, msg:CarlaEgoVehicleControl):
        with self._lock:
            target_angle = msg.steer # -1 ~ 1
            self.pid_err[:]=0
            while(True):
                force = -np.sum(self.pid_coef * self.pid_err)
                force = np.clip(force, -1, 1)
                constant = ff.Constant(int(force*(65535/2)), self.envelope)

                effect = ff.Effect(
                    ecodes.FF_CONSTANT, -1, 16384,
                    ff.Trigger(0,0),
                    ff.Replay(self.duration_ms, 0),
                    ff.EffectType(ff_constant_effect = constant)
                )

                effect_id = self.wheel.upload_effect(effect)
                self.wheel.write(ecodes.EV_FF, effect_id, 1)
                time.sleep(self.duration_ms/1000)
                self.wheel.erase_effect(effect_id)


                result = (self.wheel.absinfo(0).value-self.center)/(self.center)
                err = target_angle - result

                pre_err = self.pid_err[0]
                self.pid_err[0] = err
                self.pid_err[1] += err
                self.pid_err[2] = err - pre_err

                if abs(self.pid_err[2]) < 0.01:
                    break
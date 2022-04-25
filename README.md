## Planning and Control Stack for Autonomous Vehicle
We put planning and control stack together in one package because they are tightly coupled. The boundary between planning and control depends on the type of algorithms used.

## To test the local planner protoype
Currently I just want to see the polynominal (in frenet frame) candidates of the trajectories. The behavior is just `Behavior.NOMINAL` all the time. After this, we try `Behavior.RED_LIGHT_STOP` and `Behavior.CAR_FOLLOWING`.
```
roslaunch planning_and_control_stack base_launcher.launch
roslaunch planning_and_control_stack test_local_planner.launch
rostopic pub /carla/ego_vehicle/goal geometry_msgs/Point "x: 92.5 y: -150.0 z: 0.0"
```
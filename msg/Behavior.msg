## `behavior` is one of the flags described below.
## `data` is target speed for local planner to track if the behavior = {0, 3}.
## `data` is distance ahead for maneuver to be executed by local planner if the behavior = {1, 2, 4, 5}.
time time
uint8 behavior
float32 data

## NOMINAL: let the local planner follow the pre-determined global path
## LANE_CHANGE_TO_RIGHT: do lane change to the right
## LANE_CHANGE_TO_LEFT: do lane change to the left
## CAR_FOLLOWING: stay in the lane but possibly decrease the maximum speed
## RED_LIGHT_STOP: stop due to red light (possibly smooth stopping)
## EMERGENCY_STOP: stop due to emergency situation (possibly extreme stopping)
uint8 NOMINAL               = 0
uint8 LANE_CHANGE_TO_RIGHT  = 1
uint8 LANE_CHANGE_TO_LEFT   = 2
uint8 CAR_FOLLOWING         = 3
uint8 RED_LIGHT_STOP        = 4
uint8 EMERGENCY_STOP        = 5
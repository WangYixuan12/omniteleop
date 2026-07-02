Our policy will run at 10 Hz and WBC runs at 100 Hz.

Policy-facing 10 Hz data:
Robot state in observation should be in base frame, where observation.state.eef is "FK from measured obs/joint/{torso, arm}" and observation.state.head is obs["joint"]["head"][2] ; 
Action should be in world frame, where action.eef is "left_target, right_target given to ik.solve()", and action.head is post-clamp command sent_head[2]. 
Do not use WBC FK from result or action/joint/* as the policy EEF action.

Debug data in episode_*.hdf5 :
action/joint/* = values sent to real robot, after safety/postprocessing

Debug data in episode_*_debug.hdf5 :
debug/wbc_cmd_joint/* = WBC result.* before hardware clamp 
debug/sent_joint/* = actual sent command
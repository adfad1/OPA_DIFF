import sys
import getpass

user = getpass.getuser()

sys.path.insert(1,'/home/%s/OPA_DIFF/Environments'%user)
sys.path.insert(1,'/home/%s/OPA_DIFF/baselines'%user)
sys.path.insert(2,'/home/%s/OPA_DIFF/baselines/baselines/common'%user)
sys.path.insert(2,'/home/%s/OPA_DIFF/baselines/baselines/bench'%user)
sys.path.insert(2,'/home/%s/OPA_DIFF/baselines/baselines/ppo1'%user)



print(sys.path)
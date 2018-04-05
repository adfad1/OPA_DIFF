import sys
import getpass

user = getpass.getuser()

sys.path.insert(1,'/home/%s/OPA_DIFF/Environments'%user)
sys.path.insert(1,'/home/%s/OPA_DIFF/baselines'%user)


print(sys.path)
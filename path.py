import sys
import getpass

user = getpass.getuser()

sys.path.append('/home/%s/OPA_DIFF/Environments'%user)
sys.path.append('/home/%s/OPA_DIFF/baselines'%user)
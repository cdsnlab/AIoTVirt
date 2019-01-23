import subprocess
import json
import re

<<<<<<< HEAD
proc = subprocess.Popen(['ntpq', '-p'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf8')
=======
proc = subprocess.Popen(['ntpq', '-p'], stdout=subprocess.PIPE,stderr = subprocess.STDOUT, encoding='utf8')
>>>>>>> baf3da5ef74a42e67414534aaa0cbcf81c66c36d
stdout_value = proc.communicate()[0]

#remove the header lines
start = stdout_value.find("===\n")

if not start:
    result = {'query_result': 'failed', 'data': {}}
    print (json.dumps(result))

# Get the data part of the string
pay_dirt = stdout_value[start+4:]

# search for NTP line starting with * (primary server)
exp = ("\*((?P<remote>\S+)\s+)"
       "((?P<refid>\S+)\s+)"
       "((?P<st>\S+)\s+)"
       "((?P<t>\S+)\s+)"
       "((?P<when>\S+)\s+)"
       "((?P<poll>\S+)\s+)"
       "((?P<reach>\S+)\s+)"
       "((?P<delay>\S+)\s+)"
       "((?P<offset>\S+)\s+)"
       "((?P<jitter>\S+)\s+)")

regex = re.compile(exp, re.MULTILINE)
r = regex.search(pay_dirt)


# Did we get anything?
if not r:
    # No, try again without the * at the begining
    exp = (" ((?P<remote>\S+)\s+)"
           "((?P<refid>\S+)\s+)"
           "((?P<st>\S+)\s+)"
           "((?P<t>\S+)\s+)"
           "((?P<when>\S+)\s+)"
           "((?P<poll>\S+)\s+)"
           "((?P<reach>\S+)\s+)"
           "((?P<delay>\S+)\s+)"
           "((?P<offset>\S+)\s+)"
           "((?P<jitter>\S+)\s+)")

    regex = re.compile(exp, re.MULTILINE)
    r = regex.search(pay_dirt)

data = {}

if r:
    data = r.groupdict()

result = {'query_result': 'ok' if r else 'failed', 'data': data}

<<<<<<< HEAD
print (json.dumps(result))
=======
print (json.dumps(result))
>>>>>>> baf3da5ef74a42e67414534aaa0cbcf81c66c36d

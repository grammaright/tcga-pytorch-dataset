import os

print("aws_access_key_id:")
aws_access_key_id = input()
print("aws_secret_access_key:")
aws_secret_access_key = input()
os.system("mkdir {}/.aws".format(os.path.expanduser('~')))
f = open(os.path.expanduser('~')+"/.aws/credentials", 'w')
f.write("[default]\n")
f.write("aws_access_key_id = {}\n".format(aws_access_key_id))
f.write("aws_secret_access_key = {}\n".format(aws_secret_access_key))
f.close()

# os.system("rm -rf goofys")
# os.system("wget https://github.com/kahing/goofys/releases/latest/download/goofys")
# os.system("chmod 777 goofys")
os.system("mkdir tcga")
os.system("/Users/grammaright/Workspace/lectures/aisys-goofys/go/bin/goofys tcga-2-open tcga")

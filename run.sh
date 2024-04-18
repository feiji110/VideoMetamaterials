export RANK=0
export WORLD_SIZE=1
export MASTER_ADDR=localhost
export MASTER_PORT=12345

for ((i=1; i<=100; i ++))
do
	echo $i
    accelerate launch main.py
done
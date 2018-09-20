#!/bin/bash
echo "delete prev files"
rm data.txt
rm data1.txt
echo "measuring cpu mem usage"
cnt=0
added=0
while [ $cnt -le 10 ]
do
#	$cnt='expr $cnt+1'
	cnt=$((cnt+1))
	#echo $cnt
	# get cpu mem of process $1=>pid of the process.
	ps aux | grep -i synet | awk {'print $2'} > pid.txt
	exec < pid.txt
	while read line
	do
		ps -p $line -o %cpu >> data.txt
	done

	sleep 1
done

# now open the file and delete "%cmd" line
sed '/CPU/d' data.txt >> data1.txt
exec < data1.txt
if [ $1 -eq 0 ]; then
	while read line1
	do
		added=$line1
		echo $added
	done
elif [ $1 -eq 1 ]; then
	# read two lines and add
	while read line1 
	do
		added=0
	#	echo $line1
		read line2 
	#	echo $line2
		#added="line1+line2" | bc
		added=$(echo $line1+$line2 | bc)
		echo $added
	done
	#echo "1"
elif [ $1 -eq 2 ]; then
	# read three lines and add
	while read line1
	do
		added=0
		read line2
		read line3
		added=$(echo $line1+$line2+$line3 | bc)
		echo $added
	done
	#echo "2"
elif [ $1 -eq 3 ]; then
	# read four lines and add

	while read line1
	do
		added=0
		read line2
		read line3
		read line4
		added=$(echo $line1+$line2+$line3+$line4 | bc)
		echo $added

	done

	echo "3"
else
	while read line1
	do
		added=0
		read line2
		read line3
		read line4
		read line5
		added=$(echo $line1+$line2+$line3+$line4+$line5 | bc)
		echo $added
	done
	echo "4"
fi

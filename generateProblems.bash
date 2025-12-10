#!/bin/bash

export threads=$((16)) # <-- put how many threads you want to run here

verifyProblem () {
    local input="../tmpInstances/exp$1.txt"
    local output="../instances/test$1.txt"
    cd code
    python run_experiments.py --instance $input --solver $problemType --batch > "../logs/log$1"
    local result=$(grep -i "Total" "../logs/log$1")
    result="${result#*:}"
    # echo $result
    if [ $result ]; then
        echo "instance found"
        cp $input $output
    fi
    cd ../
    return $result
}

control_c () {
    kill $(jobs -p)
    exit
}

#---

trap control_c SIGINT
if [ "$1" = "-h" ]; then
    echo "usage: ./instanceScript.bash <#agents> <sizeX> <sizeY> <density> <#problems> <timeout>"
    exit 1
fi
if ! [ $# ]; then
    echo "usage: ./instanceScript.bash <#agents> <sizeX> <sizeY> <density> <#problems> <timeout>"
    exit 1
fi

export -f verifyProblem
export problemType=CBS

rm -f instances/* > /dev/null
rm -f tmpInstances/* > /dev/null
rm -f logs/* > /dev/null

# single thread
# for i in $(seq 1 $5); do
#     export index=$i
#     attempt=0
#     echo "generating instance $i..."
#     python3 generateProblem.py $1 $2 $3 $4 $i > /dev/null
#     echo "verifying instance $i..."
#     timeout $6 bash -c verifyProblem $i
#     while [ ! -f "instances/test$index.txt" ]; do
#         echo "instance failed"
#         echo "generating new instance $i..."
#         python3 generateProblem.py $1 $2 $3 $4 $i > /dev/null
#         echo "verifying instance $i..."
#         timeout $6 bash -c verifyProblem $i
#     done
# done

# multithread

thread () {
    i=$((0))
    echo "thread $!"
    while [ $i -le $((problems)) ]; do
        # echo $i
        if [ ! -f "tmpInstances/exp$i.txt" ]; then
            # create file lock
            touch "tmpInstances/exp$i.txt"
            while [ ! -f "instances/test$i.txt" ]; do
                python3 generateProblem.py $agents $sizeX $sizeY $density $i > /dev/null
                local input="../tmpInstances/exp$i.txt"
                local output="../instances/test$i.txt"
                cd code
                timeout $timeout python run_experiments.py --instance $input --solver $problemType --batch > "../logs/log$i"
                local result=$(grep -i "Total" "../logs/log$i")
                result="${result#*:}"
                if [ $result ]; then
                    echo "instance found $i"
                    cp $input $output
                fi
                cd ../
            done
            i=$((i+1))
        else
            i=$((i+1))
        fi
    done
}

export -f thread

export agents=$1
export sizeX=$2
export sizeY=$3
export density=$4
export problems=$5
export timeout=$6

numjobs=$( jobs | wc -l )
threads=$((threads-2))
for i in $(seq 1 $threads); do
    while [ $numjobs -le $threads ]; do
        numjobs=$( jobs | wc -l )
        thread &
        # sleep 0.1
        # echo $numjobs
    done
    wait
done
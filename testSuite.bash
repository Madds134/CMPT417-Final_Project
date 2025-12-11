#!/bin/bash

export threads=$((16)) # <-- put how many threads you want to run here

control_c () {
    kill $(jobs -p)
    exit
}

thread () {
    i=$((1))
    echo "thread $!"
    while [ $i -le $problems ]; do
        # echo $i
        if [ ! -f "logs/log$i" ]; then
            # create file lock
            touch "logs/log$i"
            echo "working on test $i"
            cd code
            timeout $timeout python run_experiments.py --instance ../${path}test$i.txt --solver $solver --macbs_low_level $type --batch > "../logs/log$i"
            # timeout 60       python run_experiments.py --instance ../instances_1020203060/test1.txt --solver MACBS --macbs_low_level joint --batch
            local result=$(grep -i "All instances complete" "../logs/log$i")
            result="${result#*:}"
            echo "result: $result"
            if [ "$result" ]; then
                echo "finished problem $i"
            else
                # did not finish flag
                echo "DNF" > ../logs/log$i
                echo "timeout on $i, did not find solution within $timeout (s)"
            fi
            cd ../
            i=$((i+1))
        else
            i=$((i+1))
        fi
    done
}

#---

export -f thread

export path=$1
export solver=$2
export problems=$3
export timeout=$4
export type=$5

trap control_c SIGINT
if [ "$1" = "-h" ]; then
    echo "usage: ./testSuite.bash <path/to/instanceFolder/> <solver> <#problems> <timeout> <type>"
    exit 1
fi
if ! [ $# ]; then
    echo "usage: ./testSuite.bash <path/to/instanceFolder/> <solver> <#problems> <timeout> <type>"
    exit 1
fi

rm logs/* > /dev/null

numjobs=$( jobs | wc -l )
threads=$((threads-2))
while [ $numjobs -le $threads ]; do
    numjobs=$( jobs | wc -l )
    thread &
    sleep 1
    echo "jobs: $numjobs"
done
wait

# rm "testData.csv" > /dev/null
echo "stitching csv"
# touch "testData.csv"

for i in $(seq 1 $problems); do
    dnfFlag=$(grep -i "DNF" "logs/log$i")
    if [ $dnfFlag ]; then
        # DNF
        # echo "-1, -1, -1, -1, test$i.txt" >> "testData.csv"
        echo "DNF"
    else
        result=$(grep -wns "All instances complete" "logs/log$i" -B 10)

        CPUtime=$(echo $result | tr ' ' '\n' | grep "CPU" -A 3)
        CPUtime=$(echo $CPUtime | tr '\n' ' ')
        CPUtime="${CPUtime#*:}"
        CPUtime="${CPUtime#*|}"

        SUMcost=$(echo $result | tr ' ' '\n' | grep "Sum" -A 3)
        SUMcost=$(echo $SUMcost | tr '\n' ' ')
        SUMcost="${SUMcost#*:}"
        SUMcost="${SUMcost#*|}"

        EXPHLnode=$(echo $result | tr ' ' '\n' | grep "Expanded" -A 3)
        EXPHLnode="${EXPHLnode#*:}"
        EXPHLnode="${EXPHLnode#*|}"

        GENHLnode=$(echo $result | tr ' ' '\n' | grep "Generated" -A 3)
        GENHLnode="${GENHLnode#*:}"
        GENHLnode="${GENHLnode#*|}"

        LLcalls=$(echo $result | tr ' ' '\n' | grep "calls:" -A 1)
        LLcalls="${LLcalls#*:}"
        LLcalls="${LLcalls#*|}"

        LLTEcall=$(echo $result | tr ' ' '\n' | grep "expansions:" -A 1)
        LLTEcall="${LLTEcall#*:}"
        LLTEcall="${LLTEcall#*|}"
        LLTEcall=$(echo $LLTEcall | tr '\n' ' ')

        LLAEcall=$(echo $result | tr ' ' '\n' | grep "call:" -A 1)
        LLAEcall="${LLAEcall#*:}"
        LLAEcall="${LLAEcall#*|}"
        LLAEcall=$(echo $LLAEcall | tr '\n' ' ')

        LLPOsize=$(echo $result | tr ' ' '\n' | grep "size:" -A 1)
        LLPOsize="${LLPOsize#*:}"
        LLPOsize="${LLPOsize#*|}"
        LLPOsize=$(echo $LLPOsize | tr '\n' ' ')

        TOTcost=$(echo $result | tr ' ' '\n' | grep "cost:" -A 1)
        TOTcost="${TOTcost#*:}"
        TOTcost=$(echo $TOTcost | tr '\n' ' ')



        echo "${type}, ${CPUtime}, ${SUMcost}, ${EXPHLnode}, ${GENHLnode}, ${LLcalls}, ${LLTEcall}, ${LLAEcall}, ${LLPOsize}, ${TOTcost}, $i" >> "testData.csv"

        # echo $result
        # echo $resultJoint
        # echo "${CPUtime}, ${SUMcost}, ${EXPnode}, ${TOLcost}, test${i}.txt" >> "testData.csv"
    fi
done
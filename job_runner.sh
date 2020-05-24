#!/bin/bash
# scheduler
while :
do
    git pull
    CMD=$(sed -e 1$'{w/dev/stdout\n;d}' -i~ "work/job_queue.sh")
    echo $CMD > "work/job_running.sh"
    eval $CMD
    echo $CMD >> "work/jobs_done.sh"
    sleep 60
done


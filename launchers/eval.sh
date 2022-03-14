#!/bin/bash
log_dir="/exps"




coopt_eval () {
    local logdir="/exps/`basename ${1%/}`"
    local ckpt=${2}
    chmod -R 777 ${logdir}
    chmod -R 777 wandb
    
    # record videos
    /docker-entrypoint.sh runuser -u sofauser -- python -m training.viz ${logdir} -t ${ckpt} -n 1
    sleep 1
    
    # eval policy
    /docker-entrypoint.sh python -m training.eval ${logdir} -t ${ckpt} -n 5
    sleep 1
    
    # record unreduced videos
    /docker-entrypoint.sh runuser -u sofauser -- python -m training.viz ${logdir} -t ${ckpt} -n 1 --unreduced
    sleep 1
}


# check for new checkpoints to evaluate


maybe_launch_eval () {
    local dir=${1%/}
    mkdir -p "${dir}/eval_flags"
    shopt -s nullglob
    for ckpt in ${dir}/ckpts/*.pt; do
        local t=`basename "${ckpt}" .pt`
        local flag="${dir}/eval_flags/${t}"
        # rm "${flag}"
        if test ! -f ${flag}; then
            touch "${flag}"
	        echo "${dir} ${t}"
            coopt_eval "${dir}" "${t}"
        fi
    done
}


launch_eval () {
    for d in ${log_dir}/*/; do
	# echo ${d}
        maybe_launch_eval "${d}"
    done
}


trap 'exit' INT


while [ 1 ]
do
    launch_eval
    # echo 'sleeping...'
    sleep 10
done	



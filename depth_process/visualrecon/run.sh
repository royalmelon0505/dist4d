

CONFIG=$1
INDICE=$2

LOG_PATH=./logs
LOG_FILE=$LOG_PATH/$CONFIG-full.log
if [ ! -d $LOG_PATH ]; then
	mkdir -p $LOG_PATH
fi
exec > >(tee -a "$LOG_FILE")
exec 2>&1

echo ======== Start overall pipeline in $(date) ========
export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO \
python pipeline.py --config $CONFIG --indice $INDICE

echo ======== Finish overall pipeline in $(date) ========
exit
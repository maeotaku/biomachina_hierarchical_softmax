#!/bin/bash
set -e
for ARGUMENT in "$@"
do
   KEY=$(echo $ARGUMENT | cut -f1 -d=)

   KEY_LENGTH=${#KEY}
   VALUE="${ARGUMENT:$KEY_LENGTH+1}"

   export "$KEY"="$VALUE"
done
me=`basename "$0"`
MINUTES=${minutes:-60}
SLURM_PARTITION=${partition:-nukwa-debug}
RUN_TMP=${run_tmp:-/dev/shm/plantclef-2022/}
DS_DIR=${ds_dir:-/work/$USER/dataset/plantclef-2022/}
SRC_DIR=${src_dir:-/home/$USER/plantclef2022/src/}
$CONDA_ENV=${conda_env:-biomachina}
BIO_TMP=$(mktemp /tmp/biomachina-XXXXX)
echo "partition=$SLURM_PARTITION"
echo "run_tmp=$RUN_TMP"
echo "ds_dir=$DS_DIR"
echo "src_dir=$SRC_DIR"
echo "conda_env=$CONDA_ENV"
echo "minutes=$MINUTES"

if [[ ! -e "${DS_DIR}web.tar" ]]; then
    echo "[ERROR] dataset not found ${DS_DIR}web.tar"
    echo "please use $me ds_dir=/your/ds_directory/ to override directory. Default /work/$USER/dataset/plantclef-2022/"
    exit 1
fi

if [[ ! -e "${DS_DIR}trusted.tar" ]]; then
    echo "[ERROR] dataset not found ${DS_DIR}trusted.tar"
    echo "please use $me ds_dir=/your/ds_directory/ to override directory. Default /work/$USER/dataset/plantclef-2022/"
    exit 1
fi

if [[ ! -e "${SRC_DIR}main.py" ]]; then
    echo "[ERROR] source code not found in directory ${SRC_DIR}"
    echo "please use $me src_dir=/your/src/ to override directory. Default /home/$USER/plantclef2022/src/"
   exit 1
fi

cat << BIO >> $BIO_TMP
#!/bin/bash
#SBATCH --job-name=biomachina
#SBATCH --output=out_%A.txt
#SBATCH --ntasks=1

ENV_NAME=$CONDA_ENV

TARGET_ENV="\$(conda env list | sed 's/^\(\w\w*\)\s .*$/\1/' | grep \$ENV_NAME)"
echo "\$TARGET_ENV"
module load cuda/11.1
module load cudnn-cuda10/7.5.1
echo "node=\$HOSTNAME"

#Create conda and install dependencies.
source ./load_env.sh
[ -z "\$TARGET_ENV" ] && echo "creating environment: \$ENV_NAME" && conda create --name \$ENV_NAME python=3.8 && echo "conda environment created"
echo "activate env \$ENV_NAME"
conda activate \$ENV_NAME
echo "installing pytorch-lts"
conda install -S -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
echo "installing dependencies"
conda install -y -S -c conda-forge pytorch-lightning omegaconf hydra-core einops pandas scikit-learn lxml wandb

# pull tar files and extract in memory file.

if [[ ! -e "$RUN_TMP" ]]; then
    echo "creating directory $RUN_TMP"
    mkdir --parent "$RUN_TMP"
    trap "rm -rf $RUN_TMP" EXIT
    echo "extracting  ${DS_DIR}web.tar"
    tar -xf "${DS_DIR}web.tar" -C "$RUN_TMP"
    echo "extracting  ${DS_DIR}trusted.tar"
    tar -xf "${DS_DIR}trusted.tar" -C "$RUN_TMP"
fi

cd "$SRC_DIR"
python main.py -cn=simclr_resnet.yaml dataset.path="${RUN_TMP}web" batch_size=192

BIO
echo "running script $BIO_TMP"
JOB_ID=$(sbatch --parsable -c 16 --partition=$SLURM_PARTITION -t $MINUTES $BIO_TMP)
sleep 2
echo "jobId $JOB_ID"
squeue -u $USER
tail -f "out_${JOB_ID}.txt"

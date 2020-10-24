#!/bin/bash

main() {
set -e

#export HIP_DB=0xf

BUILD_DIR=build

CASE_NAME="sum"     # case name
DOES_SUBMIT=        # submit or not
NUM_NODES=1         # number of nodes
NUM_RANKS=4         # number of ranks per node
GRES_SPEC="gpu:4"   # slurm gres types and counts
EXCLUDE_NODES=

# Options
opts=$(getopt -o N:n:g: -l case:,submit -- "$@")
eval set -- "${opts}"

while true; do

case "$1" in
    "-N") NUM_NODES=$2; shift 2;;
    "-n") NUM_RANKS=$2; shift 2;;
    "-g") GRES_SPEC="$2"; shift 2;;
    "--case") CASE_NAME="$2"; shift 2;;
    "--submit") DOES_SUBMIT=ON; shift 1;;
    --|*)
        shift; break;;
esac

done

# Check the executable
CASE_EXE="$BUILD_DIR/$CASE_NAME/$CASE_NAME"
if [ ! -f $CASE_EXE ]; then
    echo "ERROR: file $CASE_EXE does not exist!"
    exit 1
fi

# Report settings
printVariables

# Submission or local run
if [ "x$DOES_SUBMIT" == "xON" ]; then

    mkdir -p log/ &> /dev/null


sbatch << END
#!/bin/bash
#SBATCH -J hybrid-mpi-hip
#SBATCH -o log/%j-N${NUM_NODES}.out
#SBATCH -p normal $EXCLUDE_NODES
#SBATCH -N $NUM_NODES
#SBATCH --tasks-per-node $NUM_RANKS
#SBATCH --gres=$GRES_SPEC
#SBATCH --export=ALL

echo "$(printVariables)"

mpirun --bind-to none $CASE_EXE
END


else
    mpirun -np $NUM_RANKS $CASE_EXE
fi

}

printVariables() {
cat << END
--------------------
    Case name      = $CASE_NAME
    Case exe       = $CASE_EXE
    Nodes          = $NUM_NODES
    GRES           = $GRES_SPEC
    Ranks per node = $NUM_RANKS

    Excluded: $EXCLUDE_NODES
--------------------
END
}


main "$@"

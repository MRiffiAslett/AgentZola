#!/bin/bash
# Script to run WhiteFox in batches to cover all 49 optimizations
# Splits optimizations into groups that fit within 8-hour SLURM windows

set -euo pipefail

PROJECT_ROOT="/vol/bitbucket/mtr25/AgentZola/WhiteFox"
CONFIG_PATH="xilo_xla/config/generator.toml"

# Define optimization batches (12 optimizations per batch to fit in 8 hours)
BATCH_1="AllGatherBroadcastReorder,AllGatherCombiner,AllGatherDecomposer,AllReduceCombiner,AllReduceFolder,AllReduceReassociate,AllReduceSimplifier,AsyncCollectiveCreator,BatchDotSimplification,Bfloat16ConversionFolding,BroadcastCanonicalizer,ChangeOpDataType"

BATCH_2="CollectivesScheduleLinearizer,ConcatForwarding,ConditionalCanonicalizer,ConvertAsyncCollectivesToSync,ConvertMover,Defuser,DotDecomposer,DotMerger,DynamicIndexSplitter,HloConstantFolding,HloCse,HloDce"

BATCH_3="HloElementTypeConverter,IdentityConvertRemoving,IdentityReshapeRemoving,LoopScheduleLinearizer,MapInliner,ReduceScatterCombiner,ReduceScatterDecomposer,ReduceScatterReassociate,ReshapeBroadcastForwarding,ReshapeReshapeForwarding,ShardingRemover,SimplifyFpConversions"

BATCH_4="SliceConcatForwarding,SliceSinker,SortSimplifier,StochasticConvertDecomposer,TopkRewriter,TransposeFolding,TreeReductionRewriter,TupleSimplifier,WhileLoopConstantSinking,WhileLoopExpensiveInvariantCodeMotion,WhileLoopInvariantCodeMotion,WhileLoopTripCountAnnotator"

BATCH_5="ZeroSizedHloElimination"

echo "Submitting WhiteFox jobs in batches..."
echo

# Submit batch 1
JOB1=$(sbatch --parsable "$PROJECT_ROOT/slurm/whitefox_slurm_batch.sh" "$BATCH_1" "batch1")
echo "Batch 1 submitted: Job ID $JOB1"

# Submit batch 2 (depends on batch 1)
JOB2=$(sbatch --parsable --dependency=afterok:$JOB1 "$PROJECT_ROOT/slurm/whitefox_slurm_batch.sh" "$BATCH_2" "batch2")
echo "Batch 2 submitted: Job ID $JOB2 (depends on $JOB1)"

# Submit batch 3 (depends on batch 2)
JOB3=$(sbatch --parsable --dependency=afterok:$JOB2 "$PROJECT_ROOT/slurm/whitefox_slurm_batch.sh" "$BATCH_3" "batch3")
echo "Batch 3 submitted: Job ID $JOB3 (depends on $JOB2)"

# Submit batch 4 (depends on batch 3)
JOB4=$(sbatch --parsable --dependency=afterok:$JOB3 "$PROJECT_ROOT/slurm/whitefox_slurm_batch.sh" "$BATCH_4" "batch4")
echo "Batch 4 submitted: Job ID $JOB4 (depends on $JOB3)"

# Submit batch 5 (depends on batch 4)
JOB5=$(sbatch --parsable --dependency=afterok:$JOB4 "$PROJECT_ROOT/slurm/whitefox_slurm_batch.sh" "$BATCH_5" "batch5")
echo "Batch 5 submitted: Job ID $JOB5 (depends on $JOB4)"

echo
echo "All batches submitted successfully!"
echo "Job chain: $JOB1 -> $JOB2 -> $JOB3 -> $JOB4 -> $JOB5"


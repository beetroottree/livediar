"""Generate train_pilot.py from upstream moshi-finetune's train.py by explicit replacements.

Every replacement asserts its anchor exists, so an upstream change fails loudly
instead of silently training the wrong thing. Output goes next to upstream so
`from finetune...` imports resolve:

    python research/pilot/build_trainer.py research/moshi_finetune_repo
    cd research/moshi_finetune_repo && torchrun --nproc-per-node 1 train_pilot.py research/pilot/pilot.yaml
"""

import sys
from pathlib import Path

REPO = Path(sys.argv[1] if len(sys.argv) > 1 else "research/moshi_finetune_repo").resolve()
HERE = Path(__file__).resolve().parent
src = (REPO / "train.py").read_text()


def rep(old, new, count=1):
    global src
    assert old in src, f"anchor not found in upstream train.py:\n{old[:120]}"
    src = src.replace(old, new, count)


# imports + hooks
rep("from moshi.models import loaders",
    "from moshi.models import loaders\n"
    f"sys.path.insert(0, {str(HERE)!r}); import pilot_hooks as PH; PH.apply_all()  # pilot: dataset + fuser patches")
rep("import dataclasses\n", "import dataclasses\nimport sys\n")

# optional init from a previous segment's merged checkpoint + conditioning state
rep('''    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,''',
    '''    if os.environ.get("PILOT_INIT_MOSHI"):                     # pilot: segment restart
        args.moshi_paths.moshi_path = os.environ["PILOT_INIT_MOSHI"]
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,''')

# conditioning module + fuser after the model is built
rep('''    model = get_fsdp_model(args, checkpoint_info)
''', '''    model = get_fsdp_model(args, checkpoint_info)
    cond_module = PH.build_conditioning(model, dim=int(lm_config["dim"]))   # pilot
    if PH.load_cond(cond_module, os.environ.get("PILOT_INIT_COND")):
        main_logger_info("pilot: loaded conditioning module from PILOT_INIT_COND")
''')

# optimizer covers the conditioning module too
rep('''    optimizer = AdamW(
        model.parameters(),''', '''    optimizer = AdamW(
        list(model.parameters()) + list(cond_module.parameters()),   # pilot''')

# per-frame condition tensors from the batch's activity
rep('''            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )
''', '''            condition_tensors = PH.make_condition(cond_module, batch.activity, model, codes.shape[-1])  # pilot
''')

# save the conditioning module with every checkpoint
rep('''            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )
''', '''            checkpointer.save_checkpoint(
                save_only_lora=not args.full_finetuning and args.save_adapters,
                dtype=param_dtype,
            )
            if get_rank() == 0:
                PH.save_cond(run_dir, state.step, cond_module)              # pilot
''')

out = REPO / "train_pilot.py"
out.write_text(src)
print(f"wrote {out} ({len(src.splitlines())} lines)")

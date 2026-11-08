"""Finite synthetic checks; no pretrained model or real T output is read."""
import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import protein_T_confirm as f


def case(q=7, length=300, label="HELIX"):
    return {"id": "SYNTH", "case_id": f"SYNTH:{label}:{q}", "sequence": "A" * length,
        "q": q, "token_q_0based": q+1, "label": label, "y": float(label == "HELIX"),
        "known_group_id": "synthetic-fresh", "component": "acc:SYNTH"}


class HookModule:
    def __init__(self): self.hooks = []
    def register_forward_hook(self, hook):
        self.hooks.append(hook)
        return SimpleNamespace(remove=lambda: self.hooks.remove(hook))


class WrapperTests(unittest.TestCase):
    def test_long_sequence_and_token_window_boundaries(self):
        config = {"hidden_size": 320, "intermediate_size": 1280, "num_hidden_layers": 6,
                  "hidden_act": "gelu", "position_embedding_type": "rotary", "max_position_embeddings": 1026}
        f.validate_token_capacity(config)
        with self.assertRaises(RuntimeError): f.validate_token_capacity(dict(config, max_position_embeddings=513))
        for length in (257, 300, 512): f.validate_case(case(length=length))
        for length in (256, 513):
            with self.assertRaises(ValueError): f.validate_case(case(length=length))
        malformed = case()
        malformed["token_q_0based"] = 7
        with self.assertRaises(ValueError): f.validate_case(malformed)

    def test_same_protein_two_distinct_single_MASK_calls_to_native_postGELU_hook(self):
        import torch
        intermediate, dense = HookModule(), HookModule()
        block = SimpleNamespace(intermediate=intermediate, output=SimpleNamespace(dense=dense))
        masks = []
        class Tokenizer:
            mask_token_id = 32
            def convert_tokens_to_ids(self, residues): return [4+f.krr.AA.index(a) for a in residues]
            def __call__(self, sequences, **kwargs):
                ids = [0] + self.convert_tokens_to_ids(sequences[0]) + [2]
                return {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1,len(ids)),dtype=torch.long)}
        class SyntheticModel:
            def __call__(self, **tokens):
                indices = torch.nonzero(tokens["input_ids"][0] == 32).flatten().tolist()
                masks.append(indices)
                n = tokens["input_ids"].shape[1]
                z = torch.zeros((1,n,1280)); z[0,indices[0],:] = indices[0]
                for hook in list(intermediate.hooks): hook(intermediate, (), z)
                for hook in list(dense.hooks): hook(dense, (), torch.zeros((1,n,320)))
                return SimpleNamespace(logits=torch.zeros((1,n,33)))
        ledger = SimpleNamespace(begin=lambda purpose,case_id: {}, end=lambda *args: None)
        runner = object.__new__(f.LongNativeRunner)
        runner.torch,runner.ledger,runner.block = torch,ledger,block
        runner.tokenizer,runner.model,runner.aa_ids = Tokenizer(),SyntheticModel(),list(range(4,24))
        first = runner.run(case(q=7))
        second = runner.run(case(q=292,label="STRAND"))
        self.assertEqual(masks, [[8],[293]])
        self.assertEqual(first["z"].shape,(1280,))
        self.assertEqual(float(first["z"][1246]),8)
        self.assertEqual(float(second["z"][1246]),293)
        self.assertEqual(intermediate.hooks,[])
        with self.assertRaises(ValueError): runner.run(case(),patch={"unit":1246,"value":0})
        self.assertEqual(len(masks),2)

    def test_forward_exact_order_count_and_model_CPU_reserve(self):
        cases = [case(),case(q=292,label="STRAND")]
        with patch.object(f,"write"):
            ledger=f.ForwardLedger(cases)
            with self.assertRaises(RuntimeError): ledger.begin(f.PURPOSE,cases[1]["case_id"])
            event=ledger.begin(f.PURPOSE,cases[0]["case_id"])
            ledger.end(event,.2,.1)
            event=ledger.begin(f.PURPOSE,cases[1]["case_id"])
            ledger.end(event,.3,.1)
            with self.assertRaises(RuntimeError): ledger.begin(f.PURPOSE,cases[0]["case_id"])
            self.assertEqual(ledger.state["sequence_forwards"],2)
            self.assertEqual(ledger.state["completed_forwards"],2)
            ledger=f.ForwardLedger(cases)
            ledger.started -= f.MAX_MODEL_CPU+1
            with self.assertRaises(RuntimeError): ledger.begin(f.PURPOSE,cases[0]["case_id"])
        self.assertEqual((f.TARGET,f.FORWARDS,f.THREADS),(1304,2608,4))
        self.assertEqual((f.MAX_MODEL_CPU,f.MAX_READOUT_CPU),(3600,3600))

    def test_CPU_watchdog_calls_terminal_stop_without_real_exit(self):
        stopped = threading.Event()
        handler = Mock(side_effect=lambda _: stopped.set())
        watch = f.CPUWatchdog("synthetic",0,handler)
        with patch.object(f.os,"_exit",side_effect=lambda _: watch.done.set()) as terminal:
            with watch: self.assertTrue(stopped.wait(2))
        handler.assert_called_once()
        terminal.assert_called_once_with(124)

    def test_long_features_keep_length_over_256_and_frozen_pair_semantics(self):
        cases=[case(length=512),case(q=504,length=512,label="STRAND")]
        pairs=[{"a_index":0,"b_index":1,"y_a":1,"group":"synth"}]
        data=f.krr.pair_data(cases,pairs,f.np.asarray([2.,.5]))
        self.assertEqual(float(f.krr.control_features(cases[0]["sequence"],7)[62]),2.)
        self.assertEqual(data["dx"].shape,(1,64))
        self.assertEqual(data["pa"].shape,(1,30))
        self.assertEqual(float(data["dj"][0]),1.5)
        self.assertEqual(int(data["pa"][0,0]),20)

    def test_saved_prediction_helper_uses_fixed_alpha_trace_without_fitting(self):
        f.OUT.mkdir(parents=True,exist_ok=True)
        rng=f.np.random.default_rng(17)
        train={"dx":rng.normal(size=(3,64)),"pa":rng.integers(0,21,size=(3,30)),
               "pb":rng.integers(0,21,size=(3,30)),"dj":rng.normal(size=3)}
        target={"dx":rng.normal(size=(2,64)),"pa":rng.integers(0,21,size=(2,30)),
                "pb":rng.integers(0,21,size=(2,30)),"dj":rng.normal(size=2)}
        scales={name:float(i+1) for i,name in enumerate(f.krr.BLOCKS)}
        alpha=f.np.asarray([.1,-.2,.3])
        with tempfile.TemporaryDirectory(dir=f.OUT,prefix="synthetic_") as folder:
            path=Path(folder)/"synthetic_final.npz"
            f.np.savez(path,**{f"train_{key}":value for key,value in train.items()},
                **{f"scale_{key}":value for key,value in scales.items()},
                **{f"{model}_alpha":alpha for model in f.krr.MODELS})
            with patch.object(f.krr,"fit_krr",side_effect=AssertionError("No fitting")):
                for model in f.krr.MODELS:
                    score,p=f.krr.predict_saved(target,model,path)
                    expected=f.krr.sum_kernel(f.krr.raw_blocks(target,train),scales,model)@alpha
                    f.np.testing.assert_allclose(score,expected)
                    f.np.testing.assert_allclose(p,f.np.clip((1+expected)/2,0,1))

    def test_fixed_four_family_unavailable_keeps_denominator_and_actual_precision(self):
        losses={model:f.np.asarray([.1,.2,.3,.4]) for model in f.krr.MODELS}
        losses["Xj"]=f.np.asarray([.09,.15,.2,.3])
        groups=f.np.asarray(["a","a","b","c"])
        contributions,ids,grouped,summary=f.inference_from_losses(losses,groups)
        self.assertEqual(contributions.shape,(4,4))
        self.assertEqual(grouped.shape,(3,4))
        self.assertEqual(summary["family_size"],4)
        self.assertEqual([e["direction"] for e in summary["endpoints"]],["greater","greater","greater","two-sided"])
        self.assertEqual(summary["endpoints"][1]["p_for_Holm"],1.)
        self.assertFalse(summary["endpoints"][1]["actual_half_width_le_0_01"])
        self.assertIsNotNone(summary["endpoints"][0]["actual_simultaneous_CI_half_width"])

    def test_model_load_forbidden_without_specific_T_release(self):
        with self.assertRaises(RuntimeError): f.LongNativeRunner("unused",None,{"status":"FROZEN"})


if __name__ == "__main__": unittest.main()

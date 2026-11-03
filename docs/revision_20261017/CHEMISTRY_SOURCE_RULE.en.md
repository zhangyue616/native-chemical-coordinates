# Public-source acquisition lock for the new chemistry lane

> English reader translation of [`CHEMISTRY_SOURCE_RULE.md`](CHEMISTRY_SOURCE_RULE.md), added for this public release. It is not a study record: the Chinese file is the record of this protocol and governs if the two differ.

2026-10-17: the research operator has selected B; this file is written before this project's first download of the public parent corpus content. It locks only the public-source acquisition and is not a permission for D development or T unsealing.

- Sole URL: `https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz`.
- HEAD as measured this time: 200; Content-Length 1,486,837,335; Last-Modified `Sun, 27 Sep 2026 07:55:20 GMT`; no ETag.
- Existing SHA256 of this public-source version: `36b683a59272b196ed7ece140903a249f7df475a06c312c963a5c761f1037f1a`. A single streaming verification is performed during this independent download; identity of source must not be determined by name alone. Last-Modified must not be passed off as a download date or as a formal PubChem release version.
- Download only from the official HTTPS to `data/revision_20261017/chemistry/source/`; the old study's original parent corpus or corpus must not be read. The old 50k exclusion uses only the plain CID/identity-hash list provided by the research operator and the source receipt.
- Write to disk and compute the SHA in 1 MiB stream chunks, with at most 3 Range resumes of the same object after interruption; a Range/length/source-date mismatch stops the process immediately, and failure evidence must not be discarded. The source data must not be decompressed, molecules must not be taken, and labels or model outputs must not be produced. Once the same object is complete, it must not be downloaded or hashed again.
- As the next step, the complete data rule must specify the new salt, standardization, old-CID and standardized-identity exclusion, the semantic anchor alignment gate, the D/T assignment and the T contact boundary before decompression and sampling are permitted. Before the scientific protocol and the D lock are complete, T labels and model outputs must not be generated/read.

This is an independent confirmation on a new implementation and does not retroactively ratify the historical v3 source-code identity. That the new molecules do not overlap with the old study data does not prove that they never appeared in pretraining.

## Transfer optimization revision (after slow-download evidence, before the first decompression)

The actual acquisition on 2026-10-17 reused the same already downloaded public-source cache and verified the SHA within the copy stream, consistent with the source contract established in this file. The copy method and actual time are recorded in SOURCE_RECEIPT, and this must not be passed off as a complete network download. This change affects only the transfer path and does not change the source bytes, the new salt, standardization, the exclusion of old data or the D/T rules.

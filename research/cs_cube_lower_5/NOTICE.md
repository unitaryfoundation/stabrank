# Provenance and notices

The runtime routines in this staging directory were written during this
research project with substantial Codex assistance. `PROVENANCE.json` records
the exact internally reviewed source and function hashes copied into the
portable modules. The imported mathematical routines were extracted verbatim;
workspace-specific imports, runtime headers and orchestration were replaced.
The new direct enlarged-group preparation and clean driver are separately
reviewable source. Historical scientific results are not runtime inputs.

No source routine from `unitaryfoundation/stabrank` is copied into these
runtime modules. The earlier representation builder only read upstream
constants by AST and did not import or execute upstream source; the portable
module includes its independently written Gaussian arithmetic and stabilizer
phase checker. Accordingly there is no removed upstream source copyright
header. Should future packaging copy upstream code, its Apache-2.0 notices
must be preserved with the relevant source and license.

The known CS-square three-term identity, the stabilizer-rank challenge and
the earlier block-lift strategy are prior work, not original claims here:
[official CS challenge](https://unitaryfoundation.github.io/stabrank/orbits/CS.html).
The affine-support/quadratic-phase characterization is described in
[Dehaene and De Moor, primary text](https://arxiv.org/html/quant-ph/0304125v1).
The analytic proof states further precise attribution. The present staging
does not claim novelty or platform acceptance from a missing search result.

The upstream repository's license is Apache License 2.0 at the reviewed
[pinned commit](https://github.com/unitaryfoundation/stabrank/blob/edde871fda354479d56e49b93a2502989bd956a2/LICENSE).
This notice does not attribute the new code to the upstream project or imply
upstream review or endorsement.

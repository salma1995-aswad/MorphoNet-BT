# Dataset setup and access boundaries

Obtain the Cheng brain tumor dataset through [the official Figshare DOI](https://doi.org/10.6084/m9.figshare.1512427). Confirm the specific downloaded version, terms, attribution and permission for derivative MRI figures; availability does not establish permission for every local file. No dataset files, raw MRI, masks, patient manifests or patient-linked predictions are included here.

Set MORPHONET_DATA_ROOT to the directory containing the dataset ZIP archives/MAT files. The historical default sibling directory name was 1512427. Phase-1 inspection recursively discovers actual files, reads ZIP members directly and checks CRC. MATLAB/HDF5 records contain cjdata with label, PID, image, tumorMask and tumorBorder; auxiliary files without cjdata are separate. HDF5 MATLAB dimensions are reversed consistently for image/mask. PID/sample_id must remain strings. Class labels: 1=Meningioma→index0; 2=Glioma→1; 3=Pituitary→2. No raw IDs are provided as examples.

Expected analyzed cohort: 3,064 slices/233 patients; 3,049 at512×512 and15 at256×256. Masks are binary/nonempty. The loader checks identifiers, labels and image hashes, retaining archive::member paths. cvind.mat is not accepted as the patient-disjoint split.

Frozen splits comprise fold_1_train.csv through fold_5_test.csv (15 files), their hash metadata and a dataset manifest. Required split schema is documented in split_manifest_schema.json; it includes PID and is intentionally not populated here. Seed42 alone cannot recover an identical split without identical data ordering, library behavior and canonical records. Do not regenerate splits to claim reproduction of reported results. Local-path relocation changes CSV bytes and invalidates historical hashes; an author-reviewed portable-artifact release is required, not ad-hoc hash replacement.

MRI processing: original-resolution paired affine in Train only, then256×256 bilinear MRI and nearest-neighbor-exact masks; warpAffine masks use nearest. Per-image all-pixel population Z-score, epsilon1e-8, no clipping. Training augmentation and morphology definitions are in scientific_protocol.md; masks are internal to training preprocessing to prevent tumor clipping, not model inputs. Inference model input is MRI alone. Public notebooks are disabled by default, including preprocessing/split generation.


The exact provider release/version and downloadable byte identity are not verified by this release phase. Record them before authorized research; do not substitute a regenerated split for the canonical records. See [artifact availability](artifact_availability.md).

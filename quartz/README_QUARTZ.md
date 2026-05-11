# Quartz HPC Notes

This project used Indiana University Quartz HPC for nnU-Net training.

## 1. Local-to-Quartz sync

Replace `username` and `/N/project/YOUR_PROJECT` with the correct Quartz username and project path.

```bash
rsync -avz local_nnunet/ username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/local_nnunet/
rsync -avz quartz/slurm_jobs/ username@quartz.uits.iu.edu:/N/project/YOUR_PROJECT/slurm_jobs/

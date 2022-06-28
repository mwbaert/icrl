gpulab-cli --cert="../../../login_ilabt_imec_be_mwbaert@ugent.be.pem" --debug submit --project=phd_mattijs < $1

echo "current commit:"
git log -1

#!/bin/bash
set -euo pipefail
echo "Destroying all AWS infrastructure..."
cd "$(dirname "$0")"
terraform destroy -auto-approve
echo "All resources destroyed. No further billing."

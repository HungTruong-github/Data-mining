import zipfile
import json
import hashlib
from pathlib import Path

def test_bundle():
    zip_path = Path("review_bundle.zip")
    assert zip_path.exists(), "review_bundle.zip must exist"
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        file_list = zf.namelist()
        print(f"Total files in zip: {len(file_list)}")
        
        # Verify manifest inside
        manifest_name = "review_bundle/review_manifest.json"
        assert manifest_name in file_list, "review_manifest.json missing in zip"
        
        manifest_data = json.loads(zf.read(manifest_name).decode('utf-8'))
        print(f"Manifest status: {manifest_data.get('validation_status')}")
        print(f"Tests recorded: {manifest_data.get('tests_passed')}")
        assert manifest_data.get('validation_status') == 'PASS'
        
        # Check mandatory files
        mandatory = [
            "review_bundle/README_REVIEW.md",
            "review_bundle/tables/association_rules/association_rules_equivalence_audit.csv",
            "review_bundle/tables/classification/cv_tuning_history.csv",
            "review_bundle/tables/clustering/clustering_stability_audit.csv",
            "review_bundle/reports/pipeline_manifest.json",
            "review_bundle/validation/pytest_report.xml",
            "review_bundle/validation/acceptance_results.csv"
        ]
        for m in mandatory:
            assert m in file_list, f"Mandatory file {m} missing from zip"
            print(f"  [OK] Found in zip: {m}")
            
    print("\nALL BUNDLE VERIFICATION CHECKS PASSED!")

if __name__ == '__main__':
    test_bundle()

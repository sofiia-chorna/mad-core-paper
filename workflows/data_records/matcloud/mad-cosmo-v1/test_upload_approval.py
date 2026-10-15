import unittest

from upload import approved_names, projected_record_bytes


class ApprovalTests(unittest.TestCase):
    def setUp(self):
        self.data = 'madcosmo_cells_0-2pow18_data.h5'
        self.features = 'madcosmo_cells_0-2pow18_features.h5'
        self.detail = {'bytes': 123, 'sha256': 'a' * 64}
        self.manifest = {'files': {self.data: self.detail, self.features: self.detail}}

    def test_data_only_does_not_add_features_or_manifest(self):
        approval = {'record': '${MC_DRAFT_MADCOSMO}', 'files': {self.data: self.detail}}
        self.assertEqual(approved_names(approval, self.manifest), [self.data])

    def test_wrong_record_is_rejected(self):
        with self.assertRaises(ValueError):
            approved_names({'record': 'wrong', 'files': {self.data: self.detail}}, self.manifest)

    def test_empty_list_is_rejected(self):
        with self.assertRaises(ValueError):
            approved_names({'record': '${MC_DRAFT_MADCOSMO}', 'files': {}}, self.manifest)

    def test_unknown_file_is_rejected(self):
        with self.assertRaises(ValueError):
            approved_names({'record': '${MC_DRAFT_MADCOSMO}', 'files': {'unknown.txt': self.detail}}, self.manifest)

    def test_path_traversal_is_rejected(self):
        with self.assertRaises(ValueError):
            approved_names({'record': '${MC_DRAFT_MADCOSMO}', 'files': {'../data.h5': self.detail}}, self.manifest)

    def test_unpinned_file_is_rejected(self):
        with self.assertRaises(ValueError):
            approved_names({'record': '${MC_DRAFT_MADCOSMO}', 'files': {self.data: {'bytes': 123}}}, self.manifest)

    def test_capacity_includes_existing_unselected_files(self):
        entries = [{'key': 'README.md', 'size': 10, 'status': 'completed'}]
        self.assertEqual(projected_record_bytes(entries, {self.data: ('sha', 'md5', 90)}, 100), 100)
        with self.assertRaises(ValueError):
            projected_record_bytes(entries, {self.data: ('sha', 'md5', 91)}, 100)

    def test_existing_completed_file_is_not_counted_twice(self):
        entries = [{'key': self.data, 'size': 90, 'checksum': 'md5:abc', 'status': 'completed'}]
        self.assertEqual(projected_record_bytes(entries, {self.data: ('sha', 'abc', 90)}, 100), 90)

    def test_unknown_pending_file_is_rejected(self):
        with self.assertRaises(ValueError):
            projected_record_bytes([{'key': 'unknown', 'status': 'pending'}], {self.data: ('sha', 'abc', 90)}, 100)

    def test_changed_remote_file_is_rejected_before_upload(self):
        entries = [{'key': self.data, 'size': 90, 'checksum': 'md5:wrong', 'status': 'completed'}]
        with self.assertRaises(ValueError):
            projected_record_bytes(entries, {self.data: ('sha', 'abc', 90)}, 100)


if __name__ == '__main__':
    unittest.main()

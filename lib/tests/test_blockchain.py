import unittest
from lib.blockchain import *
#    get_header_length,
#    get_header_size_between,
#    _calculate_offset_for_header_at
#)

class TestUtil(unittest.TestCase):

    def test_get_header_length(self):
        self.assertEqual(get_header_length(-1), 0)
        self.assertEqual(get_header_length(-1234), 0)
        self.assertEqual(get_header_length(0), HDR_LEN_GENESIS)
        self.assertEqual(get_header_length(2), HDR_LEN_EPOCH_1)
        self.assertEqual(get_header_length(1234), HDR_LEN_EPOCH_1)
        self.assertEqual(get_header_length(EPOCH_1_END_BLOCK_HEIGHT), HDR_LEN_EPOCH_1)
        self.assertEqual(get_header_length(EPOCH_2_START_BLOCK_HEIGHT), HDR_LEN_EPOCH_2)
        self.assertEqual(get_header_length(EPOCH_2_START_BLOCK_HEIGHT + 2), HDR_LEN_EPOCH_2)
        self.assertEqual(get_header_length(EPOCH_2_START_BLOCK_HEIGHT + 1234), HDR_LEN_EPOCH_2)

    def test_get_header_size_between(self):
        # Negative and boundary value tests
        self.assertEqual(get_header_size_between(0, 0), 0)
        self.assertEqual(get_header_size_between(1, 1), 0)
        self.assertEqual(get_header_size_between(5, 5), 0)
        self.assertEqual(get_header_size_between(1234, 1234), 0)
        self.assertEqual(get_header_size_between(-1, 0), 0)
        self.assertEqual(get_header_size_between(0, -1), 0)
        self.assertRaises(Exception, get_header_size_between, -1, -1)
        self.assertRaises(Exception, get_header_size_between, -1, -5)
        self.assertRaises(Exception, get_header_size_between, -5, -1)

        self.assertEqual(get_header_size_between(0, 1), HDR_LEN_GENESIS)
        self.assertEqual(get_header_size_between(0, 2), HDR_LEN_GENESIS + HDR_LEN_EPOCH_1)
        self.assertEqual(get_header_size_between(0, 5), HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * 4)
        self.assertEqual(get_header_size_between(2, 3), HDR_LEN_EPOCH_1)
        self.assertEqual(get_header_size_between(3, 5), HDR_LEN_EPOCH_1 * 2)

        # Value on bounds of EPOCHS tests
        self.assertEqual(get_header_size_between(EPOCH_1_END_BLOCK_HEIGHT, EPOCH_2_START_BLOCK_HEIGHT),
                         HDR_LEN_EPOCH_1)
        self.assertEqual(get_header_size_between(EPOCH_1_END_BLOCK_HEIGHT - 1, EPOCH_2_START_BLOCK_HEIGHT),
                         HDR_LEN_EPOCH_1 * 2)
        self.assertEqual(get_header_size_between(EPOCH_1_END_BLOCK_HEIGHT - 5, EPOCH_2_START_BLOCK_HEIGHT),
                         HDR_LEN_EPOCH_1 * 6)
        self.assertEqual(get_header_size_between(EPOCH_1_END_BLOCK_HEIGHT - 1, EPOCH_2_START_BLOCK_HEIGHT + 1),
                         HDR_LEN_EPOCH_1 * 2 + HDR_LEN_EPOCH_2)
        self.assertEqual(get_header_size_between(EPOCH_1_END_BLOCK_HEIGHT - 100, EPOCH_2_START_BLOCK_HEIGHT + 1),
                         HDR_LEN_EPOCH_1 * 101 + HDR_LEN_EPOCH_2)
        self.assertEqual(get_header_size_between(EPOCH_1_END_BLOCK_HEIGHT - 100, EPOCH_2_START_BLOCK_HEIGHT + 100),
                         HDR_LEN_EPOCH_1 * 101 + HDR_LEN_EPOCH_2 * 100)

    def test_calculate_offset_for_header_at(self):
        # Negative and boundary value tests
        self.assertEqual(calculate_offset_for_header_at(0), 0)
        self.assertEqual(calculate_offset_for_header_at(-1), 0)
        self.assertEqual(calculate_offset_for_header_at(-1234), 0)

        self.assertEqual(calculate_offset_for_header_at(1), HDR_LEN_GENESIS)
        self.assertEqual(calculate_offset_for_header_at(2), HDR_LEN_GENESIS + HDR_LEN_EPOCH_1)
        self.assertEqual(calculate_offset_for_header_at(3), HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * 2)
        self.assertEqual(calculate_offset_for_header_at(100), HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * 99)

        # Value on bounds of EPOCHS tests
        self.assertEqual(calculate_offset_for_header_at(EPOCH_1_END_BLOCK_HEIGHT),
                         HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * (EPOCH_1_END_BLOCK_HEIGHT - 1))
        self.assertEqual(calculate_offset_for_header_at(EPOCH_1_END_BLOCK_HEIGHT - 100),
                         HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * (EPOCH_1_END_BLOCK_HEIGHT - 101))
        self.assertEqual(calculate_offset_for_header_at(EPOCH_2_START_BLOCK_HEIGHT),
                         HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * EPOCH_1_END_BLOCK_HEIGHT)
        self.assertEqual(calculate_offset_for_header_at(EPOCH_2_START_BLOCK_HEIGHT + 1),
                         HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * EPOCH_1_END_BLOCK_HEIGHT + HDR_LEN_EPOCH_2)
        self.assertEqual(calculate_offset_for_header_at(EPOCH_2_START_BLOCK_HEIGHT + 100),
                         HDR_LEN_GENESIS + HDR_LEN_EPOCH_1 * EPOCH_1_END_BLOCK_HEIGHT + HDR_LEN_EPOCH_2 * 100)



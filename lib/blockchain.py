# Electrum - lightweight Bitcoin client
# Copyright (C) 2012 thomasv@ecdsa.org
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import os
import threading

from . import util
from . import bitcoin
from . import constants
from .bitcoin import *

#TEMP
###
import binascii
hfu = binascii.hexlify
###

HDR_LEN_GENESIS = 141
HDR_LEN_EPOCH_1 = 1487
HDR_LEN_EPOCH_2 = 241
HDR_LEN_COMMON = HDR_LEN_EPOCH_1
# Equihash parameters (144, 5)
EPOCH_1_END_BLOCK_HEIGHT = 160010
# In the original BTCZ the epoch2 start block height is 160000. I added +10 to prevent overlapping as mined blocks use
# epoch 1 Equihash parameters (200, 9) indeed
EPOCH_2_START_BLOCK_HEIGHT = 160011
CHUNK_LEN = 100

MAX_TARGET = 0x0007FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF
POW_AVERAGING_WINDOW = 13
POW_MEDIAN_BLOCK_SPAN = 11
POW_MAX_ADJUST_DOWN = 34
POW_MAX_ADJUST_UP = 34
POW_DAMPING_FACTOR = 4
POW_TARGET_SPACING = 150

TARGET_CALC_BLOCKS = POW_AVERAGING_WINDOW + POW_MEDIAN_BLOCK_SPAN

AVERAGING_WINDOW_TIMESPAN = POW_AVERAGING_WINDOW * POW_TARGET_SPACING

MIN_ACTUAL_TIMESPAN = AVERAGING_WINDOW_TIMESPAN * \
    (100 - POW_MAX_ADJUST_UP) // 100

MAX_ACTUAL_TIMESPAN = AVERAGING_WINDOW_TIMESPAN * \
    (100 + POW_MAX_ADJUST_DOWN) // 100


def serialize_header(res, extend_with_zeros=False):
    solution = res.get('solution')
    height = res.get('block_height')
    # Extend with zeros headers of 2nd epoch to store headers correctly
    if extend_with_zeros:
        solution += '00' * (HDR_LEN_EPOCH_1 - get_header_length(height))
    s = int_to_hex(res.get('version'), 4) \
        + rev_hex(res.get('prev_block_hash')) \
        + rev_hex(res.get('merkle_root')) \
        + rev_hex(res.get('reserved_hash')) \
        + int_to_hex(int(res.get('timestamp')), 4) \
        + int_to_hex(int(res.get('bits')), 4) \
        + rev_hex(res.get('nonce')) \
        + rev_hex(res.get('sol_size')) \
        + bh2u(bfh(solution))
        #+ rev_hex(res.get('solution'))
    return s

def deserialize_header(s, height):
    if not s:
        raise Exception('Invalid header: {}'.format(s))

    real_header_length = get_header_length(height)

    # We can receive a block to deserialize from both electrum server and db
    if len(s) != HDR_LEN_COMMON and len(s) != real_header_length:
        raise Exception('Invalid header length: {}'.format(len(s)))

    hex_to_int = lambda s: int('0x' + bh2u(s[::-1]), 16)
    h = {}
    h['version'] = hex_to_int(s[0:4])
    h['prev_block_hash'] = hash_encode(s[4:36])
    h['merkle_root'] = hash_encode(s[36:68])
    h['reserved_hash'] = hash_encode(s[68:100])
    h['timestamp'] = hex_to_int(s[100:104])
    h['bits'] = hex_to_int(s[104:108])
    h['nonce'] = hash_encode(s[108:140])
    if height >= EPOCH_2_START_BLOCK_HEIGHT:
        h['sol_size'] = hash_encode(s[140:141])
        h['solution'] = rev_hex(hash_encode(s[141:real_header_length]))
    else:
        h['sol_size'] = hash_encode(s[140:143])
        h['solution'] = rev_hex(hash_encode(s[143:real_header_length]))
    h['block_height'] = height

    return h

def hash_header(header):
    if header is None:
        return '0' * 64
    if header.get('prev_block_hash') is None:
        header['prev_block_hash'] = '00'*32

    return hash_encode(Hash(bfh(serialize_header(header))))

def get_header_length(height):
    if height >= EPOCH_2_START_BLOCK_HEIGHT:
        return HDR_LEN_EPOCH_2
    if height > 0:
        return HDR_LEN_EPOCH_1
    if height == 0:
        return HDR_LEN_GENESIS

    return 0

def get_header_size_between(height1, height2):
    if height1 < 0 and height2 < 0:
        raise Exception("Can't find header size between two negative heights")
    #TODO We need to think if we should normalize these negative height values. If we normalize, the size between blocks will be 0
    if height1 < 0:
        height1 = 0
    if height2 < 0:
        height2 = 0

    diff = abs(height1 - height2)
    size = 0
    if diff == 0:
        return 0
    if height1 == 0 or height2 == 0:
        diff -= 1
        size += HDR_LEN_GENESIS

    min_height = min(height1, height2)
    max_height = max(height1, height2)

    # all blocks between are in the same epochs so we can just multiply diff on num blocks
    if (height1 >= EPOCH_2_START_BLOCK_HEIGHT and height2 >= EPOCH_2_START_BLOCK_HEIGHT) \
        or (height1 <= EPOCH_1_END_BLOCK_HEIGHT and height2 <= EPOCH_1_END_BLOCK_HEIGHT):
        return size + diff * get_header_length(max_height)

    size = 0
    for i in range(min_height, max_height):
        size += get_header_length(i)
    return size

def calculate_offset_for_header_at(height):
    if height == 0:
        return 0
    return get_header_size_between(0, height)


blockchains = {}

def read_blockchains(config):
    blockchains[0] = Blockchain(config, 0, None)
    fdir = os.path.join(util.get_headers_dir(config), 'forks')
    if not os.path.exists(fdir):
        os.mkdir(fdir)
    l = filter(lambda x: x.startswith('fork_'), os.listdir(fdir))
    l = sorted(l, key = lambda x: int(x.split('_')[1]))
    for filename in l:
        checkpoint = int(filename.split('_')[2])
        parent_id = int(filename.split('_')[1])
        b = Blockchain(config, checkpoint, parent_id)
        h = b.read_header(b.checkpoint)
        if b.parent().can_connect(h, check_height=False):
            blockchains[b.checkpoint] = b
        else:
            util.print_error("cannot connect", filename)
    return blockchains

def check_header(header):
    if type(header) is not dict:
        return False
    for b in blockchains.values():
        if b.check_header(header):
            return b
    return False

def can_connect(header):
    for b in blockchains.values():
        if b.can_connect(header):
            return b
    return False


class Blockchain(util.PrintError):
    """
    Manages blockchain headers and their verification
    """

    def __init__(self, config, checkpoint, parent_id):
        self.config = config
        self.catch_up = None # interface catching up
        self.checkpoint = checkpoint
        self.checkpoints = constants.net.CHECKPOINTS
        self.parent_id = parent_id
        self.lock = threading.Lock()
        with self.lock:
            self.update_size()

    def parent(self):
        return blockchains[self.parent_id]

    def get_max_child(self):
        children = list(filter(lambda y: y.parent_id==self.checkpoint, blockchains.values()))
        return max([x.checkpoint for x in children]) if children else None

    def get_checkpoint(self):
        mc = self.get_max_child()
        return mc if mc is not None else self.checkpoint

    def get_branch_size(self):
        return self.height() - self.get_checkpoint() + 1

    def get_name(self):
        return self.get_hash(self.get_checkpoint()).lstrip('00')[0:10]

    def check_header(self, header):
        header_hash = hash_header(header)
        height = header.get('block_height')
        return header_hash == self.get_hash(height)

    def fork(parent, header):
        checkpoint = header.get('block_height')
        self = Blockchain(parent.config, checkpoint, parent.checkpoint)
        open(self.path(), 'w+').close()
        self.save_header(header)
        return self

    def height(self):
        return self.checkpoint + self.size() - 1

    def size(self):
        with self.lock:
            return self._size

    #TODO Size could be truncated at some moment so we need to think if this is the best strategy to estimate the size
    def estimate_current_size(self):
        height = self.height()
        if height > EPOCH_1_END_BLOCK_HEIGHT:
            epoch_1_size = EPOCH_1_END_BLOCK_HEIGHT * HDR_LEN_EPOCH_1
            height_diff = height - EPOCH_1_END_BLOCK_HEIGHT
            return epoch_1_size + height_diff * HDR_LEN_EPOCH_2
        return height * HDR_LEN_EPOCH_1

    def update_size(self):
        p = self.path()
        self._size = os.path.getsize(p)//HDR_LEN_COMMON if os.path.exists(p) else 0

    def verify_header(self, header, prev_hash, target):
        _hash = hash_header(header)
        if prev_hash != header.get('prev_block_hash'):
            raise Exception("prev hash mismatch: %s vs %s" % (prev_hash, header.get('prev_block_hash')))
        if constants.net.TESTNET:
            return
        bits = self.target_to_bits(target)
        if bits != header.get('bits'):
            self.print_error(f"[verify_header] bits mismatch: bits: {bits} header.bits: {header.get('bits')}")
            raise Exception("bits mismatch: %s vs %s" % (bits, header.get('bits')))
        if header['block_height'] != 0 and int('0x' + _hash, 16) > target:
            raise Exception("insufficient proof of work: %s vs target %s" % (int('0x' + _hash, 16), target))

    def verify_chunk(self, index, data):
        num_headers = CHUNK_LEN
        prev_hash = self.get_hash(index * CHUNK_LEN - 1)
        data_cursor_pos = 0
        chunk_headers = {'empty': True}
        for i in range(num_headers):
            height = index * CHUNK_LEN + i
            header_length = get_header_length(height)
            raw_header = data[data_cursor_pos:data_cursor_pos + header_length]
            if data_cursor_pos + header_length > len(data):
                break
            data_cursor_pos += header_length
            header = deserialize_header(raw_header, height)
            target = self.get_target(height, chunk_headers)
            self.verify_header(header, prev_hash, target)

            chunk_headers[height] = header
            if i == 0:
                chunk_headers['min_height'] = height
                chunk_headers['empty'] = False
            chunk_headers['max_height'] = height
            prev_hash = hash_header(header)

    def path(self):
        d = util.get_headers_dir(self.config)
        filename = 'blockchain_headers' if self.parent_id is None else os.path.join('forks', 'fork_%d_%d'%(self.parent_id, self.checkpoint))
        return os.path.join(d, filename)

    def modify_chunk(self, index, chunk):
        num_headers = CHUNK_LEN
        data_cursor_pos = 0
        new_data_cursor_pos = 0

        new_chunk = bytearray(HDR_LEN_COMMON * num_headers)
        for i in range(num_headers):
            height = index * CHUNK_LEN + i
            real_header_length = get_header_length(height)
            if data_cursor_pos + real_header_length > len(chunk):
                #self.print_error(f"[modify_chunk] reached the end of the chunk. Last processed header in chunk: {i-1}")
                break
            raw_header = chunk[data_cursor_pos:data_cursor_pos + real_header_length]
            new_chunk[new_data_cursor_pos:new_data_cursor_pos + real_header_length] = raw_header

            data_cursor_pos += real_header_length
            new_data_cursor_pos += HDR_LEN_COMMON
        # Trim all redundant empty data (if the chunk is smaller than CHUNK_LEN)
        new_chunk = new_chunk[:new_data_cursor_pos]
        #self.print_error(f"[modify_chunk] Finished. Num headers: {i} Old size: {len(chunk)} New size: {len(new_chunk)}")
        return new_chunk

    def save_chunk(self, index, chunk):
        filename = self.path()
        chunk = self.modify_chunk(index, chunk)
        d = (index * CHUNK_LEN - self.checkpoint) * HDR_LEN_COMMON
        if d < 0:
            chunk = chunk[-d:]
            d = 0
        truncate = index >= len(self.checkpoints)
        self.write(chunk, d, truncate)
        self.swap_with_parent()

    def swap_with_parent(self):
        if self.parent_id is None:
            return
        parent_branch_size = self.parent().height() - self.checkpoint + 1
        if parent_branch_size >= self.size():
            return
        self.print_error("swap", self.checkpoint, self.parent_id)
        parent_id = self.parent_id
        checkpoint = self.checkpoint
        parent = self.parent()
        with open(self.path(), 'rb') as f:
            my_data = f.read()
        with open(parent.path(), 'rb') as f:
            #TODO Check swap with parents
            #f.seek((checkpoint - parent.checkpoint)*header_length)
            #parent_data = f.read(parent_branch_size*header_length)
            f.seek(get_header_size_between(checkpoint, parent.checkpoint))
            parent_data = f.read(get_header_size_between(self.parent().height(), self.checkpoint + 1))

        self.write(parent_data, 0)
        #TODO Check swap with parents
        #parent.write(my_data, (checkpoint - parent.checkpoint)*header_length)
        parent.write(my_data, get_header_size_between(checkpoint, parent.checkpoint))
        # store file path
        for b in blockchains.values():
            b.old_path = b.path()
        # swap parameters
        self.parent_id = parent.parent_id; parent.parent_id = parent_id
        self.checkpoint = parent.checkpoint; parent.checkpoint = checkpoint
        self._size = parent._size; parent._size = parent_branch_size
        # move files
        for b in blockchains.values():
            if b in [self, parent]: continue
            if b.old_path != b.path():
                self.print_error("renaming", b.old_path, b.path())
                os.rename(b.old_path, b.path())
        # update pointers
        blockchains[self.checkpoint] = self
        blockchains[parent.checkpoint] = parent

    def write(self, data, offset, truncate=True):
        filename = self.path()
        with self.lock:
            with open(filename, 'rb+') as f:
                if truncate and offset != self._size * HDR_LEN_EPOCH_1:
                    f.seek(offset)
                    f.truncate()
                f.seek(offset)
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            self.update_size()

    def save_header(self, header):
        height = header.get('block_height')
        delta = height - self.checkpoint
        data = bfh(serialize_header(header, extend_with_zeros=True))
        assert delta == self.size()
        header_length = HDR_LEN_COMMON
        assert len(data) == header_length
        self.write(data, delta * header_length)
        #self.write(data, calculate_offset_for_header_at(height))
        self.swap_with_parent()

    def read_header(self, height):
        assert self.parent_id != self.checkpoint
        if height < 0:
            return
        if height < self.checkpoint:
            return self.parent().read_header(height)
        if height > self.height():
            return
        delta = height - self.checkpoint
        header_length = HDR_LEN_COMMON
        name = self.path()
        if os.path.exists(name):
            with open(name, 'rb') as f:
                f.seek(delta * header_length)
                h = f.read(header_length)
                if len(h) < header_length:
                    raise Exception('Expected to read a full header. This was only {} bytes'.format(len(h)))
        elif not os.path.exists(util.get_headers_dir(self.config)):
            raise Exception('Electrum datadir does not exist. Was it deleted while running?')
        else:
            raise Exception('Cannot find headers file but datadir is there. Should be at {}'.format(name))
        if h == bytes([0])*header_length:
            return None
        return deserialize_header(h, height)

    def get_hash(self, height):
        if height == -1:
            return '0000000000000000000000000000000000000000000000000000000000000000'
        elif height == 0:
            return constants.net.GENESIS
        elif height < len(self.checkpoints) * CHUNK_LEN - TARGET_CALC_BLOCKS:
            assert (height+1) % CHUNK_LEN == 0, height
            index = height // CHUNK_LEN
            h, t, extra_headers = self.checkpoints[index]
            return h
        else:
            return hash_header(self.read_header(height))

    def get_median_time(self, height, chunk_headers=None):
        if chunk_headers is None or chunk_headers['empty']:
            chunk_empty = True
        else:
            chunk_empty = False
            min_height = chunk_headers['min_height']
            max_height = chunk_headers['max_height']

        height_range = range(max(0, height - POW_MEDIAN_BLOCK_SPAN),
                             max(1, height))
        median = []
        for h in height_range:
            header = self.read_header(h)
            if not header and not chunk_empty \
                and min_height <= h <= max_height:
                    header = chunk_headers[h]
            if not header:
                raise Exception("Can not read header at height %s" % h)
            median.append(header.get('timestamp'))

        median.sort()
        return median[len(median)//2];

    def get_target(self, height, chunk_headers=None):
        if chunk_headers is None or chunk_headers['empty']:
            chunk_empty = True
        else:
            chunk_empty = False
            min_height = chunk_headers['min_height']
            max_height = chunk_headers['max_height']

        if height <= POW_AVERAGING_WINDOW:
            return MAX_TARGET
        # Reset the difficulty after the algo fork
        if height > EPOCH_1_END_BLOCK_HEIGHT and height < EPOCH_1_END_BLOCK_HEIGHT + POW_AVERAGING_WINDOW + 1:
            return MAX_TARGET

        height_range = range(max(0, height - POW_AVERAGING_WINDOW),
                             max(1, height))
        mean_target = 0
        for h in height_range:
            header = self.read_header(h)
            if not header and not chunk_empty \
                and min_height <= h <= max_height:
                    header = chunk_headers[h]
            if not header:
                raise Exception("Can not read header at height %s" % h)
            mean_target += self.bits_to_target(header.get('bits'))
        mean_target //= POW_AVERAGING_WINDOW

        actual_timespan = self.get_median_time(height, chunk_headers) - \
            self.get_median_time(height - POW_AVERAGING_WINDOW, chunk_headers)
        actual_timespan = AVERAGING_WINDOW_TIMESPAN + \
            int((actual_timespan - AVERAGING_WINDOW_TIMESPAN) / \
                POW_DAMPING_FACTOR)
        if actual_timespan < MIN_ACTUAL_TIMESPAN:
            actual_timespan = MIN_ACTUAL_TIMESPAN
        elif actual_timespan > MAX_ACTUAL_TIMESPAN:
            actual_timespan = MAX_ACTUAL_TIMESPAN

        next_target = mean_target // AVERAGING_WINDOW_TIMESPAN * actual_timespan

        if next_target > MAX_TARGET:
            next_target = MAX_TARGET

        return next_target

    def bits_to_target(self, bits):
        bitsN = (bits >> 24) & 0xff
        if not (bitsN >= 0x03 and bitsN <= 0x1f):
            if not constants.net.TESTNET:
                raise Exception("First part of bits should be in [0x03, 0x1f]")
        bitsBase = bits & 0xffffff
        if not (bitsBase >= 0x8000 and bitsBase <= 0x7fffff):
            raise Exception("Second part of bits should be in [0x8000, 0x7fffff]")
        return bitsBase << (8 * (bitsN-3))

    def target_to_bits(self, target):
        c = ("%064x" % target)[2:]
        while c[:2] == '00' and len(c) > 6:
            c = c[2:]
        bitsN, bitsBase = len(c) // 2, int('0x' + c[:6], 16)
        if bitsBase >= 0x800000:
            bitsN += 1
            bitsBase >>= 8
        return bitsN << 24 | bitsBase

    def can_connect(self, header, check_height=True):
        if header is None:
            return False
        height = header['block_height']
        if check_height and self.height() != height - 1:
            return False
        if height == 0:
            return hash_header(header) == constants.net.GENESIS
        try:
            prev_hash = self.get_hash(height - 1)
        except:
            return False
        if prev_hash != header.get('prev_block_hash'):
            return False
        target = self.get_target(height)
        try:
            self.verify_header(header, prev_hash, target)
        except BaseException as e:
            return False
        return True

    def connect_chunk(self, idx, hexdata):
        try:
            data = bfh(hexdata)
            self.verify_chunk(idx, data)
            self.save_chunk(idx, data)
            return True
        except BaseException as e:
            self.print_error('verify_chunk %d failed'%idx, str(e))
            return False

    def get_checkpoints(self):
        # for each chunk, store the hash of the last block and the target after the chunk
        cp = []
        n = self.height() // CHUNK_LEN
        for index in range(n):
            height = (index + 1) * CHUNK_LEN - 1
            h = self.get_hash(height)
            target = self.get_target(height)
            if len(h.strip('0')) == 0:
                raise Exception('%s file has not enough data.' % self.path())
            extra_headers = []
            if os.path.exists(self.path()):
                with open(self.path(), 'rb') as f:
                    lower_header = height - TARGET_CALC_BLOCKS
                    for height in range(height, lower_header-1, -1):
                        header_length = HDR_LEN_COMMON
                        f.seek(height * header_length)
                        #f.seek(calculate_offset_for_header_at(height))
                        hd = f.read(header_length)
                        if len(hd) < header_length:
                            raise Exception(
                                'Expected to read a full header.'
                                ' This was only {} bytes'.format(len(hd)))
                        extra_headers.append((height, bh2u(hd)))
            cp.append((h, target, extra_headers))
        return cp

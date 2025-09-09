import struct

# 模拟原始数据 (假设这8个字节代表current_machine_position)
original_bytes = (4637997187424142020).to_bytes(8, 'little')

# 小端解析 (当前方式)
little_result = struct.unpack('<Q', original_bytes)[0]
print(f"小端解析: {little_result}")  # 4637997187424142020

# 大端解析
big_result = struct.unpack('>Q', original_bytes)[0] 
print(f"大端解析: {big_result}")     # 4919131752989213512
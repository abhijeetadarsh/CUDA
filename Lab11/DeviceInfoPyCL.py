import pyopencl as cl

def print_device_info():
	print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
	for platform in cl.get_platforms():
		print('=' * 60)
		print('Platform - Name: ' + platform.name)
		print('Platform - Vendor: ' + platform.vendor)
		print('Platform - Version: ' + platform.version)
		print('Platform - Profile: ' + platform.profile)
		
		for device in platform.get_devices():
			print('    ' + '-' * 56)
			print('    Device - Name: ' + device.name)
			print('    Device - Type: ' + cl.device_type.to_string(device.type))
			print('    Device - ComputeUnits: {0}' .format(device.max_compute_units))
			print('    Device - LocalMemory: {0:.0f} KB' .format(device.local_mem_size/1024.0))
			print('    Device - ConstantMemory: {0:.0f} KB' .format(device.max_constant_buffer_size/1024.0))
			print('    Device - GlobalMemory: {0:.0f} GB' .format(device.global_mem_size/1073741824.0))
			print('    Device - MaxBuffer/ImageSize: {0:.0f} MB' .format(device.max_mem_alloc_size/1048576.0))
			print('    Device - MaxWorkGroupSize: {0:.0f}' .format(device.max_work_group_size))
		
	print('\n')

if __name__ == "__main__":
	print_device_info()



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
		
	print('\n')

if __name__ == "__main__":
	print_device_info()



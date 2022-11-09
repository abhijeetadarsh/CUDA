import pyopencl as cl

def print_device_info():
	print('\n' + '=' * 60 + '\nOpenCL Platforms and Devices')
	for platform in cl.get_platforms():
		print('=' * 60)
		print('Platform - Name: ' + platform.name)
		print('Platform - Vendor: ' + platform.vendor)
		print('Platform - Version: ' + platform.version)
		print('Platform - Profile: ' + platform.profile)
		
		for device in platform.get device():
			print('

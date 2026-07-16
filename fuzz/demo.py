from address_parser import AddressParser

parser = AddressParser()

address = "266/2A Bach Đằng, P24, Bình Thạnh, TP. Hồ Chí Minh"

print(parser.process(address))

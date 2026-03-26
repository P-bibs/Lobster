import sys

def main():
	if len(sys.argv) < 3:
		print("Usage: python parse_tables.py <in_file> <out_file>")
		sys.exit(1)
	in_file = sys.argv[1]
	out_file = sys.argv[2]

	with open(in_file, 'r') as f:
		lines = f.readlines()

	tables = []
	for i, line in enumerate(lines):
		if not line.startswith("\t"):
			continue
		if "schema" in line:
			continue
		if len(line.strip()) == 0:
			tables.append([])
			continue
		print(f"Found table on line {i}")
		facts = []
		for segment in line.split("), ["):
			segment = segment.strip().replace(")", "")
			tag_body, tuple_body = segment.split("::(")

			values = tuple_body.split(", ")
			tup = tuple(map(int, values))

			# trim '[' and ']'
			tag = list(map(int,tag_body[1:-1].strip().split(" ")))
			facts.append((tag, tup))
		tables.append(facts)

	print(f"Parsed {len(tables)} tables")
	print("Lengths:")
	print([len(table) for table in tables])

	with open(out_file, 'w') as f:
		for table in tables:
			for (tag, tup) in table:
				tag = ",".join(map(str, tag))
				tup = ",".join(map(str, tup))
				f.write(f"{tag}::{tup};")
			f.write("\n")

main()


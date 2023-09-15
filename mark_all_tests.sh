#!/bin/bash


# Step 1: Identify Target Files
files=$(find typed_python/ -name "*_test.py")

# Step 2: Iterate Over Each File
for file in $files; do
  # Make a backup before modifying
  cp "$file" "${file}.bak"

  # Step 3 and 4: Random Logic for Grouping and Apply Text Transformation
  awk '
  {
    if ($0 ~ /^[ \t]*def test_/) {
      match($0, /^[ \t]*/);
      indent = substr($0, RSTART, RLENGTH);
      srand();
      rand_num = int(rand() * 2 + 1);
      if (rand_num == 1) {
        print indent "@pytest.mark.group_one";
      } else {
        print indent "@pytest.mark.group_two";
      }
    }
    print $0;
  }
  ' "$file" > "${file}.tmp" && mv "${file}.tmp" "$file"
done

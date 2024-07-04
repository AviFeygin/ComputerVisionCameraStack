# Mygrep

echo "Please enter file to search:"
read filename

echo "Please enter search key:"
read search_key

grep "$search_key" "$filename"

exit_value=$?

echo "the exit value was " $exit_value

============================================================================================================

# mygrepArg
if [ "$#" -lt 2 ]; then
    echo "Error! usage: $0 filename pattern"
    exit 1
fi

filename="$1"
pattern="$2"

if [ ! -e "$filename" ]; then
   echo "Error! \"$filename\" is not an existing file in the current directory"
   exit 1
fi

echo "There are $# command line arguments: $1 $2"

grep "$pattern" "$filename"

=====================================================================================

# mygrepArg2
if [ "$#" -lt 2 ]; then
    echo "Error! usage: $0 filename pattern"
    exit 1
fi

filename="$1"
pattern="$2"

if [ ! -e "$filename" ]; then
   echo "Error! \"$filename\" is not an existing file in the current directory"
   exit 1
fi

echo "There are $# command line arguments: $1 $2"

grep "$pattern" "$filename"

grep_exit_value=$?

if [ $grep_exit_value -ne 0 ]; then
   echo "Patterm \"$pattern\" was not found in file \"$filename\""
else

   echo "$grep_result"

   echo "Pattern \"$pattern\" was found in file \"$filename\""

fi

=====================================================================================

# mygreparg3
if [ "$#" -lt 2 ]; then
    echo "Error! usage: $0 filename pattern"
    exit 1
fi

filename="$1"
pattern="$2"

if [ ! -e "$filename" ]; then
   echo "Error! \"$filename\" is not an existing file in the current directory"
   exit 1
fi

echo "There are $# command line arguments: $1 $2"

grep_result=$(grep "$pattern" "$filename" 2>/dev/null)

grep_exit_value=$?

if [ $grep_exit_value -ne 0 ]; then
   echo "Patterm \"$pattern\" was not found in file \"$filename\""
else

   echo "Pattern \"$pattern\" was found in file \"$filename\""

fi

=====================================================================================

# numbers

while true; do

   echo "Enter a number or 'quit':"
   read input

   if [ "$input" = "quit" ]; then
      echo "Bye bye"
      break
   fi

   if [[ "$input" =~ ^-?[0-9]+$ ]]; then

      if [ "$input" -gt 0 ]; then
         echo "$input is a positive number"
      elif [ "$input" -lt 0 ]; then
         echo "$input is a negative number"
      else
         echo "$input is zero"
      fi
    fi

done




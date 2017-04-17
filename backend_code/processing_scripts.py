"""Takes a filename and returns its extracted title, author, release date, and script as a tuple.
   Input: .txt filename
   Ouput: (title, author, release date, script) tuple of strings
"""
def process_raw_text_files(filename):
    transcript_filename = filename
    title = ""
    author = ""
    year = ""
    text = ""
    script = False   #True when we are within the book's text, False otherwise
    in_title = False #This allows us to save titles that span multiple lines in the file
    
    with open(transcript_filename) as f:
        for line in f:
            #If we are within the text of the book
            if script:
                #If we have reached the end of the book's text, stop
                if "End of the Project Gutenberg EBook" in line:
                    break
                #Otherwise, save the line
                else:
                    text = text + str(line)
            #If we found the title
            if "Title:" in line and title == "":
                in_title = True
                title = line[7:]
                title = title.rstrip()
            #If we found the author
            elif "Author:" in line and author == "":
                in_title = False
                author = line[8:]
                author = author.rstrip()
            #If we found the release date
            elif "Release Date:" in line and year == "":
                in_title = False
                y = line[14:]
                y = re.sub(r'\[.*?\]', '', y)
                year = re.findall(r"\D(\d{4})\D", y)
                year = year[0]
            #If we found where the body of the book begins
            elif "***" in line and text == "":
                script = True
                in_title = False
            #If none of these are true and we're still in the title, title spans multiple lines
            elif in_title:
                title = title + " " + str(line).lstrip()
                title = title.rstrip()
    
    return (title, author, year, text)

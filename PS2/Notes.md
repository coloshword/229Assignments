### Problem 6
- working on get_words (get a normalized list of words from a message string.)
- split a message into words, normalize them, and return the resulting list. Splitting should split on spaces, for normaliziation you should convert everything to lowercase, and return the list of words 
- get_words done

- working on create_dictionary
- create a dictionary mapping words to integer indices. This function should create a dictionary of word to indices using the provided training messages. 
- should use the provided training messages, use get_words to process each message
- ignore rare words, add words to the dictionary if they occur in at least five messages 
- input: list of strings containing SMS messages
- return: python dict mapping words to integers 

- working on transform_text:
    - takes a list of text messages into a numpy array for further processing
    - should create an np array, containing the number of times each word appears in each message
    - each row in resulting should correspond to each message
    - each column should correspond to a word 
    - we should have counts for each word 
    - so num rows should be number of messages
    - and the columns, should be size of word_dictionary

- how would i construct this?

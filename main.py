"""
Crossword Puzzle Generator
-------------------------
This script generates crossword puzzles from any given subject using OpenAI's API.
It creates two PDFs: one with the empty puzzle and clues, and another with the solution.

Usage: python main.py "your subject here"
Example: python main.py "Harry Potter"
"""

import openai
from pydantic import BaseModel
import random
import re
import time
from copy import copy as duplicate
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4, landscape
import os
from dotenv import load_dotenv
import sys
import argparse

def parse_args():
    """
    Set up command-line argument parsing.
    Allows users to specify the crossword subject when running the script.
    
    Returns:
        argparse.Namespace: Contains the parsed command-line arguments
    """
    parser = argparse.ArgumentParser(description='Generate a crossword puzzle from a given subject')
    parser.add_argument('subject', type=str, help='The subject for the crossword puzzle')
    return parser.parse_args()

# Load API key from .env file
load_dotenv()

# Create directory for storing generated PDFs
pdf_dir = "PDFs"
os.makedirs(pdf_dir, exist_ok=True)

def main():
    """
    Main function that orchestrates the crossword puzzle generation process:
    1. Gets the subject from command line
    2. Generates clues and answers using OpenAI
    3. Creates the crossword grid
    4. Generates PDFs for the puzzle and solution
    """
    args = parse_args()
    PROMPT_SUBJECT = args.subject
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

    openai.api_key = OPENAI_API_KEY

    class CrosswordData(BaseModel):
        """
        Defines the structure for crossword data received from OpenAI.
        
        Attributes:
            title (str): The title of the crossword puzzle
            words (list[str]): List of answer words
            clues (list[str]): List of corresponding clues
        """
        title: str
        words: list[str] 
        clues: list[str]

    # Make API call to OpenAI to generate crossword content
    completion = openai.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "Produce a clever crossword title and 30 high quality, professional, "
                    "sometimes humourous, crossword words and clues based on the subject. "
                    "Do not number the clues."
                ),
            },
            {
                "role": "user",
                "content": f"The subject is {PROMPT_SUBJECT}",
            },
        ],
        response_format=CrosswordData,
    )

    event = completion.choices[0].message.parsed

    # Extract data from OpenAI response
    puzzle_title = event.title.strip()
    word_data = list(zip(event.words, event.clues))

    class Word(object):
        """
        Represents a single word in the crossword puzzle.
        
        Attributes:
            word (str): The actual word, converted to lowercase with spaces removed
            clue (str): The corresponding clue for this word
            length (int): Number of letters in the word
            row (int): Row position in the grid (set when placed)
            col (int): Column position in the grid (set when placed)
            vertical (bool): True if word is placed vertically, False if horizontal
            number (int): The clue number assigned to this word
        """
        def __init__(self, word=None, clue=None):
            self.word = re.sub(r'\s+', '', word.lower())  # Remove spaces, convert to lowercase
            self.clue = clue
            self.length = len(self.word)
            self.row = None
            self.col = None
            self.vertical = None
            self.number = None

        def down_across(self):
            """Returns 'down' if word is vertical, 'across' if horizontal"""
            return 'down' if self.vertical else 'across'

        def __repr__(self):
            return self.word

    class Crossword(object):
        """
        Manages the creation and manipulation of the crossword puzzle grid.
        
        Attributes:
            cols (int): Number of columns in the grid
            rows (int): Number of rows in the grid
            empty (str): Character used for empty cells
            maxloops (int): Maximum attempts to place each word
            available_words (list): Words available for placement
            current_word_list (list): Words successfully placed in grid
            grid (list): 2D list representing the puzzle grid
        """
        def __init__(self, cols, rows, empty='-', maxloops=2000, available_words=[]):
            self.cols = cols
            self.rows = rows
            self.empty = empty
            self.maxloops = maxloops
            self.available_words = available_words
            self.randomize_word_list()
            self.current_word_list = []
            self.clear_grid()
            self.debug = 0

        def clear_grid(self):
            """Initialize an empty grid with the specified dimensions"""
            self.grid = []
            for _ in range(self.rows):
                row = [self.empty]*self.cols
                self.grid.append(row)

        def randomize_word_list(self):
            """
            Prepare words for placement:
            1. Convert to Word objects if needed
            2. Randomly shuffle the list
            3. Sort by length (longest first for better fitting)
            """
            temp_list = []
            for w in self.available_words:
                if isinstance(w, Word):
                    temp_list.append(Word(w.word, w.clue))
                else:
                    temp_list.append(Word(w[0], w[1]))
            random.shuffle(temp_list)
            temp_list.sort(key=lambda i: len(i.word), reverse=True)
            self.available_words = temp_list

        def compute_crossword(self, time_permitted=1.00, spins=2):
            """
            Main algorithm to create the crossword puzzle:
            1. Makes multiple attempts to place words
            2. Keeps the best result (most words placed)
            3. Runs until time limit or all words placed
            
            Args:
                time_permitted (float): Maximum time to spend trying
                spins (int): Number of placement attempts per iteration
            """
            copy_x = Crossword(self.cols, self.rows, self.empty, self.maxloops, self.available_words)

            count = 0
            start_time = time.time()
            while (time.time() - start_time) < time_permitted or count == 0:
                self.debug += 1
                copy_x.randomize_word_list()
                copy_x.current_word_list = []
                copy_x.clear_grid()

                x = 0
                while x < spins:
                    for w in copy_x.available_words:
                        if w not in copy_x.current_word_list:
                            copy_x.fit_and_add(w)
                    x += 1

                # Keep the best result so far
                if len(copy_x.current_word_list) > len(self.current_word_list):
                    self.current_word_list = copy_x.current_word_list
                    self.grid = copy_x.grid
                count += 1

        def suggest_coord(self, word):
            """
            Find potential coordinates where a word could be placed.
            Looks for overlapping letters with already placed words.
            
            Returns:
                list: Possible coordinates sorted by fit score
            """
            coordlist = []
            glc = -1
            for given_letter in word.word:
                glc += 1
                rowc = 0
                for row in self.grid:
                    rowc += 1
                    colc = 0
                    for cell in row:
                        colc += 1
                        if given_letter == cell:
                            # Check vertical placement
                            try:
                                if rowc - glc > 0:  # Enough space above
                                    if ((rowc - glc) + word.length) <= self.rows:  # Enough space below
                                        coordlist.append([colc, rowc-glc, 1, 0, 0])
                            except:
                                pass
                            # Check horizontal placement
                            try:
                                if colc - glc > 0:  # Enough space left
                                    if ((colc - glc) + word.length) <= self.cols:  # Enough space right
                                        coordlist.append([colc-glc, rowc, 0, 0, 0])
                            except:
                                pass
            return self.sort_coordlist(coordlist, word)

        def sort_coordlist(self, coordlist, word):
            """
            Score and sort possible word placements.
            Higher scores for placements with more letter overlaps.
            
            Returns:
                list: Sorted coordinates with their scores
            """
            new_coordlist = []
            for coord in coordlist:
                col, row, vertical = coord[0], coord[1], coord[2]
                score = self.check_fit_score(col, row, vertical, word)
                coord[4] = score
                if score > 0:
                    new_coordlist.append(coord)
            random.shuffle(new_coordlist)
            new_coordlist.sort(key=lambda i: i[4], reverse=True)
            return new_coordlist

        def fit_and_add(self, word):
            """
            Attempt to place a word in the grid:
            1. For first word, place near center
            2. For subsequent words, try to overlap with existing words
            3. Check multiple possible positions
            """
            fit = False
            count = 0
            coordlist = self.suggest_coord(word)

            while not fit and count < self.maxloops:
                if len(self.current_word_list) == 0:
                    # First word - place near center
                    vertical, col, row = random.randrange(0,2), 1, 1
                    if self.check_fit_score(col, row, vertical, word):
                        fit = True
                        self.set_word(col, row, vertical, word, force=True)
                else:
                    # Try to overlap with existing words
                    try:
                        col, row, vertical = coordlist[count][0], coordlist[count][1], coordlist[count][2]
                    except IndexError:
                        return
                    if coordlist[count][4]:
                        fit = True
                        self.set_word(col, row, vertical, word, force=True)
                count += 1

        def check_fit_score(self, col, row, vertical, word):
            """
            Check if a word fits at the given position and calculate fit score.
            Score increases with number of letter overlaps.
            Also checks surrounding cells to ensure words don't touch inappropriately.
            
            Returns:
                int: Score for the placement (0 if invalid)
            """
            if col < 1 or row < 1:
                return 0
            count, score = 1, 1
            for letter in word.word:
                try:
                    active_cell = self.get_cell(col, row)
                except IndexError:
                    return 0
                if active_cell not in (self.empty, letter):
                    return 0
                if active_cell == letter:
                    score += 1

                # Check surrounding cells
                if vertical:
                    if active_cell != letter:
                        if not self.check_if_cell_clear(col+1, row): return 0
                        if not self.check_if_cell_clear(col-1, row): return 0
                    if count == 1:
                        if not self.check_if_cell_clear(col, row-1): return 0
                    if count == len(word.word):
                        if not self.check_if_cell_clear(col, row+1): return 0
                else:
                    if active_cell != letter:
                        if not self.check_if_cell_clear(col, row-1): return 0
                        if not self.check_if_cell_clear(col, row+1): return 0
                    if count == 1:
                        if not self.check_if_cell_clear(col-1, row): return 0
                    if count == len(word.word):
                        if not self.check_if_cell_clear(col+1, row): return 0

                if vertical:
                    row += 1
                else:
                    col += 1
                count += 1
            return score

        def set_word(self, col, row, vertical, word, force=False):
            """
            Place a word in the grid at specified position.
            Updates both the grid and the word object with position info.
            """
            if force:
                word.col = col
                word.row = row
                word.vertical = vertical
                self.current_word_list.append(word)
                for letter in word.word:
                    self.set_cell(col, row, letter)
                    if vertical:
                        row += 1
                    else:
                        col += 1

        def set_cell(self, col, row, value):
            """Set the value of a cell in the grid"""
            self.grid[row-1][col-1] = value

        def get_cell(self, col, row):
            """Get the value of a cell in the grid"""
            return self.grid[row-1][col-1]

        def check_if_cell_clear(self, col, row):
            """Check if a cell is empty and within grid boundaries"""
            try:
                if self.get_cell(col, row) == self.empty:
                    return True
            except IndexError:
                pass
            return False

        def solution(self):
            """
            Create a string representation of the grid solution.
            
            Returns:
                str: Grid with placed letters and empty cells
            """
            outStr = ""
            for r in range(self.rows):
                for c in self.grid[r]:
                    outStr += c + " "
                outStr += "\n"
            return outStr

    # Create the crossword puzzle
    puzzle = Crossword(
        cols=20,
        rows=20,
        empty='-',
        maxloops=5000,
        available_words=word_data
    )
    puzzle.compute_crossword(time_permitted=2)

    def number_crossword_squares(puzzle):
        """
        Numbers squares according to standard crossword convention:
        - Numbers increment from left to right, top to bottom
        - Only number squares that start at least one word
        - Each square gets only one number, even if it starts both across and down words
        """
        numbering = {}
        next_num = 1
        
        # Scan grid from top to bottom, left to right
        for r in range(puzzle.rows):
            for c in range(puzzle.cols):
                # Skip empty squares
                if puzzle.grid[r][c] == puzzle.empty:
                    continue
                    
                # Check if this square starts any word
                starts_word = False
                
                # Check if it starts an across word
                is_across_start = (
                    # Must have empty square or grid edge to left
                    (c == 0 or puzzle.grid[r][c-1] == puzzle.empty) and
                    # Must have at least one letter to right
                    (c < puzzle.cols-1 and puzzle.grid[r][c+1] != puzzle.empty)
                )
                
                # Check if it starts a down word
                is_down_start = (
                    # Must have empty square or grid edge above
                    (r == 0 or puzzle.grid[r-1][c] == puzzle.empty) and
                    # Must have at least one letter below
                    (r < puzzle.rows-1 and puzzle.grid[r+1][c] != puzzle.empty)
                )
                
                # If this square starts either type of word, number it
                if is_across_start or is_down_start:
                    numbering[(r, c)] = next_num
                    next_num += 1
        
        return numbering

    # Number the squares and assign numbers to words
    numbering_dict = number_crossword_squares(puzzle)
    for w in puzzle.current_word_list:
        r0 = w.row - 1
        c0 = w.col - 1
        w.number = numbering_dict[(r0, c0)]

    # Separate and sort clues into across and down
    across_clues = []
    down_clues = []
    for w in puzzle.current_word_list:
        if w.down_across() == 'across':
            across_clues.append((w.number, w.clue))
        else:
            down_clues.append((w.number, w.clue))

    across_clues.sort(key=lambda x: x[0])
    down_clues.sort(key=lambda x: x[0])

    def wrap_text(text, max_chars=52):
        """
        Simple word-wrapping function for clue text.
        Ensures lines don't exceed specified width.
        
        Args:
            text (str): Text to wrap
            max_chars (int): Maximum characters per line
            
        Returns:
            list: Lines of wrapped text
        """
        words = text.split()
        lines = []
        current_line = []
        length_so_far = 0
        for w in words:
            if length_so_far + len(w) + len(current_line) > max_chars:
                lines.append(" ".join(current_line))
                current_line = [w]
                length_so_far = len(w)
            else:
                current_line.append(w)
                length_so_far += len(w)
        if current_line:
            lines.append(" ".join(current_line))
        return lines

    def create_crossword_pdfs(
        puzzle,
        numbering,
        across_clues,
        down_clues,
        title="My Crossword Puzzle"
    ):
        """
        Create two PDF files:
        1. The puzzle with empty grid and clues
        2. The solution with filled grid and clues
        
        Both PDFs maintain identical layouts for consistency.
        """
        # Create filenames for puzzle and solution PDFs
        safe_title = re.sub(r"\s+", "_", title.strip())
        puzzle_filename = os.path.join(pdf_dir, f"{safe_title}_puzzle.pdf")
        solution_filename = os.path.join(pdf_dir, f"{safe_title}_solution.pdf")

        def draw_page(canvas_obj, include_answers=False):
            """
            Helper function to draw a complete puzzle page.
            Used for both puzzle and solution PDFs to ensure consistent layout.
            """
            page_width, page_height = landscape(A4)
            left_margin = 50
            top_margin = page_height - 50
            text_width = 360

            # Draw title
            canvas_obj.setFont("Helvetica-Bold", 16)
            canvas_obj.drawString(left_margin, top_margin, title)
            if include_answers:
                canvas_obj.drawString(left_margin + 450, top_margin, "SOLUTION")

            y_text = top_margin - 30

            # Draw ACROSS clues
            canvas_obj.setFont("Helvetica-Bold", 12)
            canvas_obj.drawString(left_margin, y_text, "Across")
            y_text -= 15
            canvas_obj.setFont("Helvetica", 8)
            for num, clue in across_clues:
                lines = wrap_text(f"{num}. {clue}", max_chars=60)
                for ln in lines:
                    canvas_obj.drawString(left_margin, y_text, ln)
                    y_text -= 10
                y_text -= 2

            y_text -= 8

            # Draw DOWN clues
            canvas_obj.setFont("Helvetica-Bold", 12)
            canvas_obj.drawString(left_margin, y_text, "Down")
            y_text -= 15
            canvas_obj.setFont("Helvetica", 8)
            for num, clue in down_clues:
                lines = wrap_text(f"{num}. {clue}", max_chars=60)
                for ln in lines:
                    canvas_obj.drawString(left_margin, y_text, ln)
                    y_text -= 10
                y_text -= 2

            # Draw the puzzle grid
            cell_size = 20
            puzzle_left = left_margin + text_width - 30  # Reduced by 60 pixels
            puzzle_top = top_margin - 30  # Moved down

            canvas_obj.setLineWidth(1)
            for r in range(puzzle.rows):
                for cc in range(puzzle.cols):
                    x = puzzle_left + cc * cell_size
                    y = puzzle_top - r * cell_size
                    
                    # Draw black squares for empty cells
                    if puzzle.grid[r][cc] == puzzle.empty:
                        canvas_obj.setFillColor('black')
                        canvas_obj.rect(x, y - cell_size, cell_size, cell_size, fill=1)
                    else:
                        canvas_obj.setFillColor('white')
                        canvas_obj.rect(x, y - cell_size, cell_size, cell_size, fill=1)
                        
                        # Add numbers if present
                        if (r, cc) in numbering:
                            canvas_obj.setFillColor('black')
                            canvas_obj.setFont("Helvetica", 6)
                            canvas_obj.drawString(x + 2, (y - cell_size) + 10, str(numbering[(r, cc)]))
                        
                        # Add letters if this is the solution page
                        if include_answers:
                            canvas_obj.setFillColor('black')
                            canvas_obj.setFont("Helvetica-Bold", 10)
                            letter = puzzle.grid[r][cc].upper()
                            canvas_obj.drawCentredString(x + cell_size/2, (y - cell_size) + cell_size/2 - 2, letter)

        # Create puzzle PDF
        c_puzzle = canvas.Canvas(puzzle_filename, pagesize=landscape(A4))
        draw_page(c_puzzle, include_answers=False)
        c_puzzle.save()

        # Create solution PDF
        c_solution = canvas.Canvas(solution_filename, pagesize=landscape(A4))
        draw_page(c_solution, include_answers=True)
        c_solution.save()

        print(f"Crossword puzzle created: {puzzle_filename}")
        print(f"Solution created: {solution_filename}")

    # Create the PDFs
    create_crossword_pdfs(
        puzzle=puzzle,
        numbering=numbering_dict,
        across_clues=across_clues,
        down_clues=down_clues,
        title=puzzle_title
    )

    # Print solution to console (optional)
    print("Crossword solution (letters placed):")
    print(puzzle.solution())

if __name__ == "__main__":
    main()
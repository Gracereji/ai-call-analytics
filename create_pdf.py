from fpdf import FPDF
import os

# create data folder if not exists
os.makedirs("data", exist_ok=True)

pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

text = """
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Machine Learning allows computers to learn from data.
Natural Language Processing helps computers understand human language.
Deep Learning uses neural networks for complex tasks.
"""

pdf.multi_cell(0, 10, text)

pdf.output("data/sample.pdf")

print("PDF created successfully!")
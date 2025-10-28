#!/usr/bin/env python3
"""
Paper Template Generator for DJ Controller
Creates printable PDF templates for DJ board layouts.
"""

import sys
import os
import numpy as np
from reportlab.lib.pagesizes import A3, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm
from reportlab.lib.colors import black, blue, red, green, yellow, orange
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


class PaperTemplateGenerator:
    """Generates printable paper templates for DJ board layouts."""
    
    def __init__(self, page_size=A3):
        """Initialize template generator."""
        self.page_size = page_size
        self.page_width, self.page_height = page_size
        
        # Template settings
        self.margin = 2 * cm
        self.usable_width = self.page_width - 2 * self.margin
        self.usable_height = self.page_height - 2 * self.margin
        
        # Colors for different zone types
        self.zone_colors = {
            'fader': blue,
            'button': red,
            'knob': green,
            'circular': green,
            'xy_pad': orange
        }
    
    def create_2deck_template(self, output_file: str = "paper_template_2deck.pdf"):
        """Create a 2-deck DJ controller paper template."""
        c = canvas.Canvas(output_file, pagesize=self.page_size)
        
        # Template dimensions (40cm x 30cm to fit A3)
        template_width = 38 * cm
        template_height = 28 * cm
        
        # Calculate scaling and positioning
        scale_x = self.usable_width / template_width
        scale_y = self.usable_height / template_height
        scale = min(scale_x, scale_y) * 0.9  # Use 90% to leave some margin
        
        scaled_width = template_width * scale
        scaled_height = template_height * scale
        
        # Center the template on page
        offset_x = self.margin + (self.usable_width - scaled_width) / 2
        offset_y = self.margin + (self.usable_height - scaled_height) / 2
        
        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(offset_x, offset_y + scaled_height + 1*cm, "DJ Controller Paper Template - 2 Deck Layout")
        
        # Draw border
        c.setStrokeColor(black)
        c.setLineWidth(2)
        c.rect(offset_x, offset_y, scaled_width, scaled_height)
        
        # Function to convert template coordinates to PDF coordinates
        def template_to_pdf(x_cm, y_cm):
            x_pdf = offset_x + (x_cm / 40.0) * scaled_width
            y_pdf = offset_y + ((30.0 - y_cm) / 30.0) * scaled_height  # Flip Y axis
            return x_pdf, y_pdf
        
        # Draw control zones
        zones = self._get_2deck_zones()
        
        for zone in zones:
            zone_type = zone['type']
            color = self.zone_colors.get(zone_type, black)
            
            c.setStrokeColor(color)
            c.setFillColor(color)
            
            if zone_type == 'circular':
                # Draw circle
                center_x, center_y = template_to_pdf(zone['center'][0], zone['center'][1])
                radius = zone['radius'] * scale * (cm / 1.0)  # Convert radius
                c.circle(center_x, center_y, radius, fill=0, stroke=1)
                
                # Add label
                c.setFont("Helvetica", 8)
                c.setFillColor(black)
                c.drawString(center_x - len(zone['name'])*2, center_y - radius - 10, zone['name'])
            
            elif zone_type == 'fader':
                # Draw rectangle
                x1, y1 = template_to_pdf(zone['bounds'][0][0], zone['bounds'][0][1])
                x2, y2 = template_to_pdf(zone['bounds'][1][0], zone['bounds'][1][1])
                
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                
                c.rect(min(x1, x2), min(y1, y2), width, height, fill=0, stroke=1)
                
                # Add label
                c.setFont("Helvetica", 8)
                c.setFillColor(black)
                label_x = min(x1, x2) + width/2 - len(zone['name'])*2
                label_y = min(y1, y2) - 12
                c.drawString(label_x, label_y, zone['name'])
            
            elif zone_type == 'button':
                # Draw circle for buttons
                center_x, center_y = template_to_pdf(zone['center'][0], zone['center'][1])
                radius = zone['radius'] * scale * (cm / 1.0)
                c.circle(center_x, center_y, radius, fill=0, stroke=1)
                
                # Add label
                c.setFont("Helvetica", 8)
                c.setFillColor(black)
                c.drawString(center_x - len(zone['name'])*2, center_y - radius - 10, zone['name'])
            
            elif zone_type == 'knob':
                # Draw circle for knobs
                center_x, center_y = template_to_pdf(zone['center'][0], zone['center'][1])
                radius = zone['radius'] * scale * (cm / 1.0)
                c.circle(center_x, center_y, radius, fill=0, stroke=1)
                
                # Draw indicator line
                c.line(center_x, center_y, center_x, center_y + radius)
                
                # Add label
                c.setFont("Helvetica", 8)
                c.setFillColor(black)
                c.drawString(center_x - len(zone['name'])*2, center_y - radius - 10, zone['name'])
        
        # Add corner markers for calibration (optional)
        marker_size = 0.5 * cm
        c.setStrokeColor(red)
        c.setLineWidth(1)
        
        # Corner squares
        corners = [
            (offset_x, offset_y + scaled_height - marker_size),  # Top-left
            (offset_x + scaled_width - marker_size, offset_y + scaled_height - marker_size),  # Top-right
            (offset_x + scaled_width - marker_size, offset_y),  # Bottom-right
            (offset_x, offset_y)  # Bottom-left
        ]
        
        for i, (x, y) in enumerate(corners):
            c.rect(x, y, marker_size, marker_size, fill=1)
            c.setFillColor(black)
            c.setFont("Helvetica", 6)
            c.drawString(x + marker_size/4, y + marker_size/4, str(i))
            c.setFillColor(red)
        
        # Add legend
        legend_y = offset_y - 2*cm
        c.setFillColor(black)
        c.setFont("Helvetica-Bold", 12)
        c.drawString(offset_x, legend_y, "Zone Types:")
        
        legend_items = [
            ("Faders", blue),
            ("Buttons", red), 
            ("Knobs/Jogs", green),
            ("XY Pads", orange)
        ]
        
        for i, (label, color) in enumerate(legend_items):
            x = offset_x + i * 4*cm
            y = legend_y - 0.5*cm
            
            c.setStrokeColor(color)
            c.setFillColor(color)
            c.rect(x, y, 0.3*cm, 0.3*cm, fill=1)
            
            c.setFillColor(black)
            c.setFont("Helvetica", 10)
            c.drawString(x + 0.5*cm, y, label)
        
        # Add instructions
        instructions = [
            "Instructions:",
            "1. Print this template on A3 paper (100% scale, no scaling)",
            "2. Cut out the template along the border",
            "3. Place on flat surface with good lighting",
            "4. Use DJ Controller software with paper template detection",
            "5. Ensure camera can see entire template clearly"
        ]
        
        instr_y = legend_y - 1.5*cm
        c.setFont("Helvetica", 10)
        for instruction in instructions:
            c.drawString(offset_x, instr_y, instruction)
            instr_y -= 0.4*cm
        
        # Add QR code area (placeholder)
        qr_x = offset_x + scaled_width - 3*cm
        qr_y = legend_y - 2*cm
        c.setStrokeColor(black)
        c.rect(qr_x, qr_y, 2*cm, 2*cm)
        c.setFont("Helvetica", 8)
        c.drawString(qr_x + 0.1*cm, qr_y - 0.3*cm, "QR Code")
        c.drawString(qr_x + 0.1*cm, qr_y - 0.6*cm, "(Template ID)")
        
        c.save()
        print(f"✅ 2-deck template created: {output_file}")
    
    def _get_2deck_zones(self):
        """Get control zone definitions for 2-deck layout."""
        return [
            # Left deck
            {
                'name': 'JOG_L',
                'type': 'circular',
                'center': [10.0, 8.0],
                'radius': 4.0
            },
            {
                'name': 'VOL_L', 
                'type': 'fader',
                'bounds': [[3.0, 15.0], [4.0, 25.0]]
            },
            {
                'name': 'EQ_H_L',
                'type': 'knob',
                'center': [6.0, 12.0],
                'radius': 1.2
            },
            {
                'name': 'EQ_M_L',
                'type': 'knob',
                'center': [6.0, 15.0],
                'radius': 1.2
            },
            {
                'name': 'EQ_L_L',
                'type': 'knob',
                'center': [6.0, 18.0],
                'radius': 1.2
            },
            {
                'name': 'PLAY_L',
                'type': 'button',
                'center': [8.0, 22.0],
                'radius': 1.5
            },
            {
                'name': 'CUE_L',
                'type': 'button',
                'center': [12.0, 22.0],
                'radius': 1.5
            },
            {
                'name': 'SYNC_L',
                'type': 'button',
                'center': [10.0, 25.0],
                'radius': 1.0
            },
            
            # Right deck
            {
                'name': 'JOG_R',
                'type': 'circular',
                'center': [30.0, 8.0],
                'radius': 4.0
            },
            {
                'name': 'VOL_R',
                'type': 'fader', 
                'bounds': [[36.0, 15.0], [37.0, 25.0]]
            },
            {
                'name': 'EQ_H_R',
                'type': 'knob',
                'center': [34.0, 12.0],
                'radius': 1.2
            },
            {
                'name': 'EQ_M_R',
                'type': 'knob',
                'center': [34.0, 15.0],
                'radius': 1.2
            },
            {
                'name': 'EQ_L_R',
                'type': 'knob',
                'center': [34.0, 18.0],
                'radius': 1.2
            },
            {
                'name': 'PLAY_R',
                'type': 'button',
                'center': [32.0, 22.0],
                'radius': 1.5
            },
            {
                'name': 'CUE_R',
                'type': 'button',
                'center': [28.0, 22.0],
                'radius': 1.5
            },
            {
                'name': 'SYNC_R',
                'type': 'button',
                'center': [30.0, 25.0],
                'radius': 1.0
            },
            
            # Center
            {
                'name': 'CROSSFADER',
                'type': 'fader',
                'bounds': [[16.0, 26.0], [24.0, 27.0]]
            },
            {
                'name': 'BROWSE',
                'type': 'knob',
                'center': [20.0, 15.0],
                'radius': 1.5
            },
            {
                'name': 'MASTER',
                'type': 'fader',
                'bounds': [[38.5, 5.0], [39.0, 12.0]]
            },
            {
                'name': 'PHONES',
                'type': 'knob',
                'center': [20.0, 5.0],
                'radius': 1.0
            }
        ]
    
    def create_4deck_template(self, output_file: str = "paper_template_4deck.pdf"):
        """Create a 4-deck DJ controller paper template."""
        # This would be similar to 2-deck but with 4 decks
        # Implementation similar to create_2deck_template but with more zones
        print("🚧 4-deck template generation not implemented yet")
    
    def create_custom_template(self, zones_config: str, output_file: str = "paper_template_custom.pdf"):
        """Create a custom template from zones configuration file."""
        try:
            with open(zones_config, 'r') as f:
                config = json.load(f)
            
            # Implementation would parse the custom zones and create template
            print(f"🚧 Custom template generation from {zones_config} not implemented yet")
            
        except Exception as e:
            print(f"❌ Failed to load zones config: {e}")


def main():
    """Main function."""
    print("DJ Controller Paper Template Generator")
    print("=" * 50)
    
    generator = PaperTemplateGenerator(page_size=A3)
    
    # Create templates
    try:
        print("📄 Generating 2-deck template...")
        generator.create_2deck_template("templates/paper_template_2deck_A3.pdf")
        
        print("📄 Generating A4 version...")
        generator_a4 = PaperTemplateGenerator(page_size=A4)
        generator_a4.create_2deck_template("templates/paper_template_2deck_A4.pdf")
        
        print("\n✅ Templates generated successfully!")
        print("\nGenerated files:")
        print("  • templates/paper_template_2deck_A3.pdf (recommended)")
        print("  • templates/paper_template_2deck_A4.pdf (compact)")
        print("\nUsage:")
        print("  1. Print templates at 100% scale (no scaling)")
        print("  2. Use good lighting and flat surface")
        print("  3. Run DJ Controller software with paper detection enabled")
        
    except Exception as e:
        print(f"❌ Failed to generate templates: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    # Create templates directory
    os.makedirs("templates", exist_ok=True)
    
    try:
        from reportlab.lib.pagesizes import A3, A4
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import cm
        from reportlab.lib.colors import black, blue, red, green, yellow, orange
        
        exit_code = main()
    except ImportError:
        print("❌ ReportLab not installed. Install with: pip install reportlab")
        print("   This is required for PDF generation.")
        exit_code = 1
    
    sys.exit(exit_code)
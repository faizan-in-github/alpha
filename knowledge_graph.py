import networkx as nx

def create_knowledge_graph():
    G = nx.DiGraph()

    nodes = [
        # Generic Service Categories
        "Internet Setup", "Carpentry Services", "Electrical Services", "Hair Styling", 
        "Landscaping", "Plumbing", "Painting Services", "Cleaning Services", "Air Conditioning",
        
        # Specific Technical Terms
        "Fiber Optic Installation", "VPN Integration", "Server Configuration", "High-Rise Painting", 
        "Eco-Friendly Paint", "Trim and Molding", "Pressure Washing", "Staining Specialist", 
        "Deck Painting", "Organic Hair Products", "Energy-Efficient Wiring", "Smart Lighting", 
        "Custom Woodwork", "Cabinetry", "Leak Detection", "Emergency Plumbing", 
        "AC Repair", "Seasonal Landscaping", "Custom Hair Color", "Keratin Treatments", 
        "Deep Cleaning", "Move-in/Move-out Cleaning", "Sustainable Lawn Care",

        # General Service Categories
        "Plumbing", "Leak Detection", "Emergency Plumbing", "Pipe Installation", "Fixture Replacement",
        
        # Plumbing Components and Pipe Types
        "15mm Pipe", "20mm Pipe", "PVC Pipe", "CPVC Pipe", "PEX Pipe", "Copper Pipe", 
        
        # Plumbing Applications
        "Hot Water Piping", "Cold Water Piping", "Drainage", "Outdoor Plumbing", "Indoor Plumbing", 
        
        # Specific Plumbing Services and Techniques
        "Leak Repair", "Water Heater Installation", "Pipe Insulation", "Corrosion Protection", 
        "Pipe Maintenance", "Soldering", "Pipe Fitting"
    ]

    G.add_nodes_from(nodes)

    edges = [
         # General Plumbing Services and Applications
        ("Plumbing", "Pipe Installation", "Includes"),
        ("Plumbing", "Leak Detection", "Includes"),
        ("Plumbing", "Fixture Replacement", "Includes"),
        ("Plumbing", "Emergency Plumbing", "Provides"),
        
        # Pipe types and their suitable applications
        ("15mm Pipe", "Hot Water Piping", "Can Be Used For"),
        ("15mm Pipe", "Cold Water Piping", "Can Be Used For"),
        ("20mm Pipe", "Hot Water Piping", "Can Be Used For"),
        ("PVC Pipe", "Cold Water Piping", "Suitable For"),
        ("PVC Pipe", "Drainage", "Suitable For"),
        ("CPVC Pipe", "Hot Water Piping", "Ideal For"),
        ("PEX Pipe", "Hot Water Piping", "Suitable For"),
        ("PEX Pipe", "Cold Water Piping", "Suitable For"),
        ("Copper Pipe", "Hot Water Piping", "Preferred For"),
        ("Copper Pipe", "Cold Water Piping", "Preferred For"),
        ("Copper Pipe", "Corrosion Protection", "Requires"),
        
        # Specific Services and Techniques Related to Plumbing
        ("Leak Repair", "Leak Detection", "Involves"),
        ("Water Heater Installation", "Pipe Fitting", "Requires"),
        ("Pipe Insulation", "Hot Water Piping", "Often Used For"),
        ("Pipe Insulation", "Cold Water Piping", "Sometimes Used For"),
        ("Pipe Installation", "Soldering", "May Require"),
        ("Pipe Installation", "Pipe Fitting", "Requires"),
        ("Pipe Installation", "Pipe Maintenance", "Includes"),
        
        # Service-Specific Terms Related to Plumbing
        ("Drainage", "Outdoor Plumbing", "Applicable To"),
        ("Drainage", "Indoor Plumbing", "Applicable To"),
        ("Fixture Replacement", "Indoor Plumbing", "Typically For"),
        ("Emergency Plumbing", "Leak Repair", "Often Includes"),
        ("Emergency Plumbing", "Water Heater Installation", "Can Include"),

        # Internet Setup and Network Infrastructure
        ("Internet Setup", "Fiber Optic Installation", "Requires"),
        ("Internet Setup", "VPN Integration", "Involves"),
        ("Internet Setup", "Server Configuration", "Requires"),

        # Carpentry and Specific Carpentry Services
        ("Carpentry Services", "Custom Woodwork", "Includes"),
        ("Carpentry Services", "Cabinetry", "Includes"),
        
        # Electrical Services and Energy Efficiency
        ("Electrical Services", "Energy-Efficient Wiring", "Includes"),
        ("Electrical Services", "Smart Lighting", "Can Include"),

        # Hair Styling and Eco-friendly/Technical Terms
        ("Hair Styling", "Organic Hair Products", "Uses"),
        ("Hair Styling", "Custom Hair Color", "Specializes In"),
        ("Hair Styling", "Keratin Treatments", "Offers"),

        # Landscaping and Specialized Tasks
        ("Landscaping", "Seasonal Landscaping", "Offers"),
        ("Landscaping", "Sustainable Lawn Care", "Uses"),

        # Plumbing Services
        ("Plumbing", "Leak Detection", "Includes"),
        ("Plumbing", "Emergency Plumbing", "Can Offer"),

        # Painting Services and Specific Technical Terms
        ("Painting Services", "High-Rise Painting", "Specializes In"),
        ("Painting Services", "Eco-Friendly Paint", "Uses"),
        ("Painting Services", "Trim and Molding", "Can Include"),
        ("Painting Services", "Pressure Washing", "Preparation Step"),
        
        # AC and Repair Services
        ("Air Conditioning", "AC Repair", "Includes"),
        
        # Cleaning Services and Specific Tasks
        ("Cleaning Services", "Deep Cleaning", "Offers"),
        ("Cleaning Services", "Move-in/Move-out Cleaning", "Can Include"),

        # Staining and Deck Painting Specialties
        ("Staining Specialist", "Deck Painting", "Specialist In")
    ]

    for src, dst, relation in edges:
        G.add_edge(src, dst, relationship=relation)

    return G

"""A script to download GeoSciML, EarthResourceML, and ICS geological time vocabularies in Turtle (TTL) format.
It skips files that already exist in the output directory and provides detailed download statistics.   """

import requests
import os
from urllib.parse import urljoin
import time
from typing import Dict, List, Tuple

def download_geosciml_vocabularies(output_dir: str = "./data/vocabularies") -> dict:
    """
    Download GeoSciML, EarthResourceML, and ICS geological time vocabularies in TTL format.
    Skips files that already exist in the output directory.
    
    Args:
        output_dir: Output directory path for TTL files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Track download results
    successful_downloads = []
    failed_downloads = []
    skipped_downloads = []
    
    # Build vocabulary name to URL mapping
    url_mapping = {}
    
    # Standard GeoSciML vocabularies based on official vocabulary list
    geosciml_base = "http://resource.geosciml.org/classifier/cgi/"
    geosciml_vocab_mappings = {
        'Alteration Type': 'alterationtype',
        'Borehole Drilling Method': 'boreholedrillingmethod',
        'Composition Category': 'compositioncategory',
        'Compound Material Constituent Part': 'compoundmaterialconstituentpartrole',
        'Consolidation Degree': 'consolidationdegree',
        'Contact Type': 'contacttype',
        'Convention for Strike and Dip Measurements': 'conventioncode',
        'Deformation Style': 'deformationstyle',
        'Description Purpose': 'descriptionpurpose',
        'Event Environment': 'eventenvironment',
        'Event Process': 'eventprocess',
        'Fault Movement Sense': 'faultmovementsense',
        'Fault Movement Type': 'faultmovementtype',
        'Fault Type': 'faulttype',
        'Foliation Type': 'foliationtype',
        'Genetic Category': 'geneticcategory',
        'Geologic Unit Morphology': 'geologicunitmorphology',
        'Geologic Unit Part Role': 'geologicunitpartrole',
        'Geologic Unit Type': 'geologicunittype',
        'Lineation Type': 'lineationtype',
        'Mapping Frame': 'mappingframe',
        'Metamorphic Facies': 'metamorphicfacies',
        'Metamorphic Grade': 'metamorphicgrade',
        'Observation Method (Geologic Feature)': 'featureobservationmethod',
        'Observation Method (Mapped Feature)': 'mappedfeatureobservationmethod',
        'Orientation Determination Method': 'determinationmethodorientation',
        'Particle Aspect Ratio': 'particleaspectratio',
        'Particle Shape': 'particleshape',
        'Particle Type': 'particletype',
        'Planar Polarity Code': 'planarpolaritycode',
        'Proportion Term': 'proportionterm',
        'Simple Lithology': 'lithology',
        'Stratigraphic Rank': 'stratigraphicrank',
        'Value Qualifier': 'valuequalifier',
        'Vocabulary Relation': 'vocabularyrelation'
    }
    
    for vocab_name, uri_name in geosciml_vocab_mappings.items():
        url_mapping[vocab_name] = geosciml_base + uri_name
    
    # EarthResourceML vocabularies
    earthresourceml_base = "http://resource.geosciml.org/classifier/cgi/"
    earthresourceml_vocab_mappings = {
        'Commodity Code': 'commodity-code',
        'Earth Resource Expression': 'earth-resource-expression',
        'Earth Resource Form': 'earth-resource-form',
        'Earth Resource Material Role': 'earth-resource-material-role',
        'Earth Resource Shape': 'earth-resource-shape',
        'End Use Potential': 'end-use-potential',
        'Environmental Impact': 'environmental-impact',
        'Exploration Activity Type': 'exploration-activity-type',
        'Exploration Result': 'exploration-result',
        'Mine Status': 'mine-status',
        'Mineral Occurrence Type': 'mineral-occurrence-type',
        'Mining Activity': 'mining-activity',
        'Processing Activity': 'mining-processing-activity',
        'Raw Material Role': 'raw-material-role',
        'Reporting Classification Method': 'classification-method-used',
        'Reserve Assessment Category': 'reserve-assessment-category',
        'Resource Assessment Category': 'resource-assessment-category',
        'UNFC Code': 'unfc',
        'Waste Storage': 'waste-storage'
    }
    
    for vocab_name, uri_name in earthresourceml_vocab_mappings.items():
        url_mapping[vocab_name] = earthresourceml_base + uri_name
    
    # ICS geological time vocabulary
    geologic_time_vocab_mappings = {
        'International Chronostratigraphic Chart - 2020': 'https://vocabs.ardc.edu.au/repository/api/lda/csiro/international-chronostratigraphic-chart/geologic-time-scale-2020/collection.ttl'
    }
    
    for vocab_name, vocab_url in geologic_time_vocab_mappings.items():
        url_mapping[vocab_name] = vocab_url
    
    # Download vocabularies
    headers = {
        'Accept': 'text/turtle',
        'User-Agent': 'Mozilla/5.0 (compatible; GeoSciML-Downloader/1.0)'
    }
    
    total_vocabularies = len(url_mapping)
    
    for vocab_name, vocab_url in url_mapping.items():
        # Create safe filename from vocabulary name
        safe_filename = vocab_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()
        output_file = os.path.join(output_dir, f"{safe_filename}.ttl")
        
        # Skip if file already exists
        if os.path.exists(output_file):
            skipped_downloads.append(vocab_name)
            continue
            
        try:
            response = requests.get(vocab_url, headers=headers, timeout=30, allow_redirects=True)
            
            if response.status_code == 200:
                content = response.text.strip()
                
                # Verify TTL format
                if (content.startswith('@prefix') or content.startswith('@base') or 
                    '@prefix' in content[:2000] or 'rdf:type' in content[:2000]):
                    
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    successful_downloads.append(vocab_name)
                else:
                    failed_downloads.append((vocab_name, "Invalid TTL format"))
            else:
                error_msg = f"HTTP {response.status_code}"
                failed_downloads.append((vocab_name, error_msg))
            
            # Small delay to be respectful to servers
            time.sleep(0.5)
            
        except Exception as e:
            error_msg = str(e)
            failed_downloads.append((vocab_name, error_msg))
    
    # Return summary statistics
    return {
        'total': total_vocabularies,
        'successful': len(successful_downloads),
        'skipped': len(skipped_downloads),
        'failed': len(failed_downloads),
        'successful_downloads': successful_downloads,
        'skipped_downloads': skipped_downloads,
        'failed_downloads': failed_downloads,
        'output_dir': os.path.abspath(output_dir)
    }
        
        
if __name__ == "__main__":
    result = download_geosciml_vocabularies(output_dir="./data/vocabularies")
    
    print(f"Download Summary:")
    print(f"Total vocabularies: {result['total']}")
    print(f"Successfully downloaded: {result['successful']}")
    print(f"Skipped (already exist): {result['skipped']}")
    print(f"Failed: {result['failed']}")
    print(f"Output directory: {result['output_dir']}")
    
    if result['failed'] > 0:
        print("\nFailed downloads:")
        for vocab, error in result['failed_downloads']:
            print(f"  - {vocab}: {error}")
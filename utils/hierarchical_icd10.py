"""
Hierarchical ICD-10 processing with graph-based matching
"""

import re
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class ICD10Node:
    """Node in the ICD-10 hierarchy tree"""
    code: str
    description: str
    level: int  # 0=chapter, 1=block, 2=category, 3=subcategory
    parent: Optional['ICD10Node'] = None
    children: List['ICD10Node'] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def add_child(self, child: 'ICD10Node'):
        """Add a child node"""
        child.parent = self
        child.level = self.level + 1
        self.children.append(child)
    
    def get_path_to_root(self) -> List[str]:
        """Get path from this node to root"""
        path = []
        current = self
        while current:
            path.append(current.code)
            current = current.parent
        return path[::-1]
    
    def get_all_ancestors(self) -> Set[str]:
        """Get all ancestor codes"""
        ancestors = set()
        current = self.parent
        while current:
            ancestors.add(current.code)
            current = current.parent
        return ancestors
    
    def get_all_descendants(self) -> Set[str]:
        """Get all descendant codes"""
        descendants = set()
        for child in self.children:
            descendants.add(child.code)
            descendants.update(child.get_all_descendants())
        return descendants

class HierarchicalICD10Matcher:
    """Graph-based hierarchical ICD-10 matcher"""
    
    def __init__(self):
        self.root = ICD10Node("ROOT", "ICD-10 Root", 0)
        self.code_map: Dict[str, ICD10Node] = {}
        self.chapter_map: Dict[str, str] = {}
        self.block_map: Dict[str, str] = {}
        self._build_hierarchy()
    
    def _build_hierarchy(self):
        """Build comprehensive ICD-10 hierarchy"""
        
        # Chapter A: Certain infectious and parasitic diseases (A00-B99)
        chapter_a = ICD10Node("A00-B99", "Certain infectious and parasitic diseases", 1)
        self.root.add_child(chapter_a)
        
        # Block A00-A09: Intestinal infectious diseases
        block_a00 = ICD10Node("A00-A09", "Intestinal infectious diseases", 2)
        chapter_a.add_child(block_a00)
        
        # Categories and subcategories
        cat_a09 = ICD10Node("A09", "Infectious gastroenteritis and colitis, unspecified", 3)
        block_a00.add_child(cat_a09)
        subcat_a09_0 = ICD10Node("A09.0", "Other and unspecified gastroenteritis", 4)
        subcat_a09_9 = ICD10Node("A09.9", "Gastroenteritis, unspecified", 4)
        cat_a09.add_child(subcat_a09_0)
        cat_a09.add_child(subcat_a09_9)
        
        # Chapter I: Diseases of the circulatory system (I00-I99)
        chapter_i = ICD10Node("I00-I99", "Diseases of the circulatory system", 1)
        self.root.add_child(chapter_i)
        
        # Block I10-I15: Hypertensive diseases
        block_i10 = ICD10Node("I10-I15", "Hypertensive diseases", 2)
        chapter_i.add_child(block_i10)
        
        # Category I10: Essential hypertension
        cat_i10 = ICD10Node("I10", "Essential (primary) hypertension", 3)
        block_i10.add_child(cat_i10)
        subcat_i10_0 = ICD10Node("I10.0", "Malignant essential hypertension", 4)
        subcat_i10_1 = ICD10Node("I10.1", "Benign essential hypertension", 4)
        subcat_i10_9 = ICD10Node("I10.9", "Essential hypertension, unspecified", 4)
        cat_i10.add_child(subcat_i10_0)
        cat_i10.add_child(subcat_i10_1)
        cat_i10.add_child(subcat_i10_9)
        
        # Block I20-I25: Ischaemic heart diseases
        block_i20 = ICD10Node("I20-I25", "Ischaemic heart diseases", 2)
        chapter_i.add_child(block_i20)
        
        # Category I20: Angina pectoris
        cat_i20 = ICD10Node("I20", "Angina pectoris", 3)
        block_i20.add_child(cat_i20)
        subcat_i20_0 = ICD10Node("I20.0", "Unstable angina", 4)
        subcat_i20_1 = ICD10Node("I20.1", "Angina pectoris with documented spasm", 4)
        subcat_i20_8 = ICD10Node("I20.8", "Other forms of angina pectoris", 4)
        subcat_i20_9 = ICD10Node("I20.9", "Angina pectoris, unspecified", 4)
        cat_i20.add_child(subcat_i20_0)
        cat_i20.add_child(subcat_i20_1)
        cat_i20.add_child(subcat_i20_8)
        cat_i20.add_child(subcat_i20_9)
        
        # Category I21: Acute myocardial infarction
        cat_i21 = ICD10Node("I21", "Acute myocardial infarction", 3)
        block_i20.add_child(cat_i21)
        subcat_i21_0 = ICD10Node("I21.0", "Acute transmural myocardial infarction of anterior wall", 4)
        subcat_i21_1 = ICD10Node("I21.1", "Acute transmural myocardial infarction of inferior wall", 4)
        subcat_i21_9 = ICD10Node("I21.9", "Acute myocardial infarction, unspecified", 4)
        cat_i21.add_child(subcat_i21_0)
        cat_i21.add_child(subcat_i21_1)
        cat_i21.add_child(subcat_i21_9)
        
        # Chapter J: Diseases of the respiratory system (J00-J99)
        chapter_j = ICD10Node("J00-J99", "Diseases of the respiratory system", 1)
        self.root.add_child(chapter_j)
        
        # Block J40-J47: Chronic lower respiratory diseases
        block_j40 = ICD10Node("J40-J47", "Chronic lower respiratory diseases", 2)
        chapter_j.add_child(block_j40)
        
        # Category J44: Other chronic obstructive pulmonary disease
        cat_j44 = ICD10Node("J44", "Other chronic obstructive pulmonary disease", 3)
        block_j40.add_child(cat_j44)
        subcat_j44_0 = ICD10Node("J44.0", "COPD with acute lower respiratory infection", 4)
        subcat_j44_1 = ICD10Node("J44.1", "COPD with acute exacerbation", 4)
        subcat_j44_9 = ICD10Node("J44.9", "COPD, unspecified", 4)
        cat_j44.add_child(subcat_j44_0)
        cat_j44.add_child(subcat_j44_1)
        cat_j44.add_child(subcat_j44_9)
        
        # Category J45: Asthma
        cat_j45 = ICD10Node("J45", "Asthma", 3)
        block_j40.add_child(cat_j45)
        subcat_j45_0 = ICD10Node("J45.0", "Predominantly allergic asthma", 4)
        subcat_j45_1 = ICD10Node("J45.1", "Nonallergic asthma", 4)
        subcat_j45_8 = ICD10Node("J45.8", "Mixed asthma", 4)
        subcat_j45_9 = ICD10Node("J45.9", "Asthma, unspecified", 4)
        cat_j45.add_child(subcat_j45_0)
        cat_j45.add_child(subcat_j45_1)
        cat_j45.add_child(subcat_j45_8)
        cat_j45.add_child(subcat_j45_9)
        block_j40.add_child(cat_j44)
        subcat_j44_0 = ICD10Node("J44.0", "COPD with acute lower respiratory infection", 4)
        subcat_j44_1 = ICD10Node("J44.1", "COPD with acute exacerbation", 4)
        subcat_j44_9 = ICD10Node("J44.9", "COPD, unspecified", 4)
        cat_j44.add_child(subcat_j44_0)
        cat_j44.add_child(subcat_j44_1)
        cat_j44.add_child(subcat_j44_9)
        
        # Chapter E: Endocrine, nutritional and metabolic diseases (E00-E89)
        chapter_e = ICD10Node("E00-E89", "Endocrine, nutritional and metabolic diseases", 1)
        self.root.add_child(chapter_e)
        
        # Block E10-E14: Diabetes mellitus
        block_e10 = ICD10Node("E10-E14", "Diabetes mellitus", 2)
        chapter_e.add_child(block_e10)
        
        # Category E11: Non-insulin-dependent diabetes mellitus
        cat_e11 = ICD10Node("E11", "Non-insulin-dependent diabetes mellitus", 3)
        block_e10.add_child(cat_e11)
        subcat_e11_0 = ICD10Node("E11.0", "NIDDM with coma", 4)
        subcat_e11_1 = ICD10Node("E11.1", "NIDDM with ketoacidosis", 4)
        subcat_e11_9 = ICD10Node("E11.9", "NIDDM without complications", 4)
        cat_e11.add_child(subcat_e11_0)
        cat_e11.add_child(subcat_e11_1)
        cat_e11.add_child(subcat_e11_9)
        
        # Chapter M: Diseases of the musculoskeletal system (M00-M99)
        chapter_m = ICD10Node("M00-M99", "Diseases of the musculoskeletal system", 1)
        self.root.add_child(chapter_m)
        
        # Block M20-M25: Other joint disorders
        block_m20 = ICD10Node("M20-M25", "Other joint disorders", 2)
        chapter_m.add_child(block_m20)
        
        # Category M25: Other joint disorder, not elsewhere classified
        cat_m25 = ICD10Node("M25", "Other joint disorder, not elsewhere classified", 3)
        block_m20.add_child(cat_m25)
        subcat_m25_5 = ICD10Node("M25.5", "Pain in joint", 3)
        cat_m25.add_child(subcat_m25_5)
        subcat_m25_51 = ICD10Node("M25.51", "Pain in shoulder", 4)
        subcat_m25_52 = ICD10Node("M25.52", "Pain in elbow", 4)
        subcat_m25_5.add_child(subcat_m25_51)
        subcat_m25_5.add_child(subcat_m25_52)
        
        # Chapter R: Symptoms, signs and abnormal findings (R00-R99)
        chapter_r = ICD10Node("R00-R99", "Symptoms, signs and abnormal findings", 1)
        self.root.add_child(chapter_r)
        
        # Block R00-R09: Symptoms and signs involving the circulatory and respiratory systems
        block_r00 = ICD10Node("R00-R09", "Symptoms and signs involving the circulatory and respiratory systems", 2)
        chapter_r.add_child(block_r00)
        
        # Category R04: Hemorrhage from respiratory passages
        cat_r04 = ICD10Node("R04", "Hemorrhage from respiratory passages", 3)
        block_r00.add_child(cat_r04)
        subcat_r04_0 = ICD10Node("R04.0", "Epistaxis", 4)
        subcat_r04_1 = ICD10Node("R04.1", "Hemorrhage from throat", 4)
        subcat_r04_2 = ICD10Node("R04.2", "Hemoptysis", 4)
        subcat_r04_8 = ICD10Node("R04.8", "Hemorrhage from other sites in respiratory passages", 4)
        subcat_r04_9 = ICD10Node("R04.9", "Hemorrhage from respiratory passages, unspecified", 4)
        cat_r04.add_child(subcat_r04_0)
        cat_r04.add_child(subcat_r04_1)
        cat_r04.add_child(subcat_r04_2)
        cat_r04.add_child(subcat_r04_8)
        cat_r04.add_child(subcat_r04_9)
        
        # Block R40-R46: Symptoms and signs involving cognition, perception, emotional state
        block_r40 = ICD10Node("R40-R46", "Symptoms and signs involving cognition, perception, emotional state", 2)
        chapter_r.add_child(block_r40)
        
        # Category R42: Dizziness and giddiness
        cat_r42 = ICD10Node("R42", "Dizziness and giddiness", 3)
        block_r40.add_child(cat_r42)
        
        # Category R41: Other symptoms and signs involving cognitive functions
        cat_r41 = ICD10Node("R41", "Other symptoms and signs involving cognitive functions", 3)
        block_r40.add_child(cat_r41)
        subcat_r41_0 = ICD10Node("R41.0", "Disorientation, unspecified", 4)
        subcat_r41_1 = ICD10Node("R41.1", "Anterograde amnesia", 4)
        subcat_r41_3 = ICD10Node("R41.3", "Other amnesia", 4)
        cat_r41.add_child(subcat_r41_0)
        cat_r41.add_child(subcat_r41_1)
        cat_r41.add_child(subcat_r41_3)
        
        # Block R50-R69: General symptoms and signs
        block_r50 = ICD10Node("R50-R69", "General symptoms and signs", 2)
        chapter_r.add_child(block_r50)
        
        # Category R69: Illness, unspecified
        cat_r69 = ICD10Node("R69", "Illness, unspecified", 3)
        block_r50.add_child(cat_r69)
        
        # Category R51: Headache
        cat_r51 = ICD10Node("R51", "Headache", 3)
        block_r50.add_child(cat_r51)
        
        # Chapter G: Diseases of the nervous system (G00-G99)
        chapter_g = ICD10Node("G00-G99", "Diseases of the nervous system", 1)
        self.root.add_child(chapter_g)
        
        # Block G20-G26: Extrapyramidal and movement disorders
        block_g20 = ICD10Node("G20-G26", "Extrapyramidal and movement disorders", 2)
        chapter_g.add_child(block_g20)
        
        # Category G25: Other extrapyramidal and movement disorders
        cat_g25 = ICD10Node("G25", "Other extrapyramidal and movement disorders", 3)
        block_g20.add_child(cat_g25)
        subcat_g25_0 = ICD10Node("G25.0", "Essential tremor", 4)
        subcat_g25_1 = ICD10Node("G25.1", "Drug-induced tremor", 4)
        subcat_g25_2 = ICD10Node("G25.2", "Other specified forms of tremor", 4)
        subcat_g25_3 = ICD10Node("G25.3", "Myoclonus", 4)
        cat_g25.add_child(subcat_g25_0)
        cat_g25.add_child(subcat_g25_1)
        cat_g25.add_child(subcat_g25_2)
        cat_g25.add_child(subcat_g25_3)
        
        # Block G40-G47: Episodic and paroxysmal disorders
        block_g40 = ICD10Node("G40-G47", "Episodic and paroxysmal disorders", 2)
        chapter_g.add_child(block_g40)
        
        # Category G43: Migraine
        cat_g43 = ICD10Node("G43", "Migraine", 3)
        block_g40.add_child(cat_g43)
        subcat_g43_0 = ICD10Node("G43.0", "Migraine without aura", 4)
        subcat_g43_1 = ICD10Node("G43.1", "Migraine with aura", 4)
        subcat_g43_9 = ICD10Node("G43.9", "Migraine, unspecified", 4)
        cat_g43.add_child(subcat_g43_0)
        cat_g43.add_child(subcat_g43_1)
        cat_g43.add_child(subcat_g43_9)
        
        # Category G44: Other headache syndromes
        cat_g44 = ICD10Node("G44", "Other headache syndromes", 3)
        block_g40.add_child(cat_g44)
        subcat_g44_0 = ICD10Node("G44.0", "Cluster headaches", 4)
        subcat_g44_1 = ICD10Node("G44.1", "Vascular headache", 4)
        subcat_g44_2 = ICD10Node("G44.2", "Tension-type headache", 4)
        cat_g44.add_child(subcat_g44_0)
        cat_g44.add_child(subcat_g44_1)
        cat_g44.add_child(subcat_g44_2)
        
        # Chapter L: Diseases of the skin and subcutaneous tissue (L00-L99)
        chapter_l = ICD10Node("L00-L99", "Diseases of the skin and subcutaneous tissue", 1)
        self.root.add_child(chapter_l)
        
        # Block L20-L30: Dermatitis and eczema
        block_l20 = ICD10Node("L20-L30", "Dermatitis and eczema", 2)
        chapter_l.add_child(block_l20)
        
        # Category L29: Pruritus
        cat_l29 = ICD10Node("L29", "Pruritus", 3)
        block_l20.add_child(cat_l29)
        subcat_l29_0 = ICD10Node("L29.0", "Pruritus ani", 4)
        subcat_l29_1 = ICD10Node("L29.1", "Pruritus scroti", 4)
        subcat_l29_2 = ICD10Node("L29.2", "Pruritus vulvae", 4)
        subcat_l29_3 = ICD10Node("L29.3", "Anogenital pruritus, unspecified", 4)
        subcat_l29_8 = ICD10Node("L29.8", "Other pruritus", 4)
        subcat_l29_9 = ICD10Node("L29.9", "Pruritus, unspecified", 4)
        cat_l29.add_child(subcat_l29_0)
        cat_l29.add_child(subcat_l29_1)
        cat_l29.add_child(subcat_l29_2)
        cat_l29.add_child(subcat_l29_3)
        cat_l29.add_child(subcat_l29_8)
        cat_l29.add_child(subcat_l29_9)
        
        # Category L30: Other and unspecified dermatitis
        cat_l30 = ICD10Node("L30", "Other and unspecified dermatitis", 3)
        block_l20.add_child(cat_l30)
        subcat_l30_0 = ICD10Node("L30.0", "Nummular dermatitis", 4)
        subcat_l30_1 = ICD10Node("L30.1", "Dyshidrosis [pompholyx]", 4)
        subcat_l30_2 = ICD10Node("L30.2", "Cutaneous autosensitization", 4)
        subcat_l30_3 = ICD10Node("L30.3", "Infective dermatitis", 4)
        subcat_l30_4 = ICD10Node("L30.4", "Erythema intertrigo", 4)
        subcat_l30_5 = ICD10Node("L30.5", "Pityriasis alba", 4)
        subcat_l30_8 = ICD10Node("L30.8", "Other specified dermatitis", 4)
        subcat_l30_9 = ICD10Node("L30.9", "Dermatitis, unspecified", 4)
        cat_l30.add_child(subcat_l30_0)
        cat_l30.add_child(subcat_l30_1)
        cat_l30.add_child(subcat_l30_2)
        cat_l30.add_child(subcat_l30_3)
        cat_l30.add_child(subcat_l30_4)
        cat_l30.add_child(subcat_l30_5)
        cat_l30.add_child(subcat_l30_8)
        cat_l30.add_child(subcat_l30_9)
        
        # Block L40-L45: Papulosquamous disorders
        block_l40 = ICD10Node("L40-L45", "Papulosquamous disorders", 2)
        chapter_l.add_child(block_l40)
        
        # Category L40: Psoriasis
        cat_l40 = ICD10Node("L40", "Psoriasis", 3)
        block_l40.add_child(cat_l40)
        subcat_l40_0 = ICD10Node("L40.0", "Psoriasis vulgaris", 4)
        subcat_l40_1 = ICD10Node("L40.1", "Generalized pustular psoriasis", 4)
        subcat_l40_9 = ICD10Node("L40.9", "Psoriasis, unspecified", 4)
        cat_l40.add_child(subcat_l40_0)
        cat_l40.add_child(subcat_l40_1)
        cat_l40.add_child(subcat_l40_9)
        
        # Chapter K: Diseases of the digestive system (K00-K95)
        chapter_k = ICD10Node("K00-K95", "Diseases of the digestive system", 1)
        self.root.add_child(chapter_k)
        
        # Block K20-K31: Diseases of esophagus, stomach and duodenum
        block_k20 = ICD10Node("K20-K31", "Diseases of esophagus, stomach and duodenum", 2)
        chapter_k.add_child(block_k20)
        
        # Category K21: Gastro-esophageal reflux disease
        cat_k21 = ICD10Node("K21", "Gastro-esophageal reflux disease", 3)
        block_k20.add_child(cat_k21)
        subcat_k21_0 = ICD10Node("K21.0", "GERD with esophagitis", 4)
        subcat_k21_9 = ICD10Node("K21.9", "GERD without esophagitis", 4)
        cat_k21.add_child(subcat_k21_0)
        cat_k21.add_child(subcat_k21_9)
        
        # Block K50-K64: Diseases of intestines
        block_k50 = ICD10Node("K50-K64", "Diseases of intestines", 2)
        chapter_k.add_child(block_k50)
        
        # Category K59: Other functional intestinal disorders
        cat_k59 = ICD10Node("K59", "Other functional intestinal disorders", 3)
        block_k50.add_child(cat_k59)
        subcat_k59_0 = ICD10Node("K59.0", "Constipation", 4)
        subcat_k59_1 = ICD10Node("K59.1", "Diarrhea", 4)
        cat_k59.add_child(subcat_k59_0)
        cat_k59.add_child(subcat_k59_1)
        
        # Block K35-K37: Diseases of appendix
        block_k35 = ICD10Node("K35-K37", "Diseases of appendix", 2)
        chapter_k.add_child(block_k35)
        
        # Category K35: Acute appendicitis
        cat_k35 = ICD10Node("K35", "Acute appendicitis", 3)
        block_k35.add_child(cat_k35)
        subcat_k35_0 = ICD10Node("K35.0", "Acute appendicitis with generalized peritonitis", 4)
        subcat_k35_1 = ICD10Node("K35.1", "Acute appendicitis with peritoneal abscess", 4)
        subcat_k35_2 = ICD10Node("K35.2", "Acute appendicitis with generalized peritonitis, without mention of complications", 4)
        subcat_k35_3 = ICD10Node("K35.3", "Acute appendicitis with localized peritonitis", 4)
        subcat_k35_8 = ICD10Node("K35.8", "Other and unspecified acute appendicitis", 4)
        subcat_k35_9 = ICD10Node("K35.9", "Acute appendicitis, unspecified", 4)
        cat_k35.add_child(subcat_k35_0)
        cat_k35.add_child(subcat_k35_1)
        cat_k35.add_child(subcat_k35_2)
        cat_k35.add_child(subcat_k35_3)
        cat_k35.add_child(subcat_k35_8)
        cat_k35.add_child(subcat_k35_9)
        
        # Chapter N: Diseases of the genitourinary system (N00-N99)
        chapter_n = ICD10Node("N00-N99", "Diseases of the genitourinary system", 1)
        self.root.add_child(chapter_n)
        
        # Block N30-N39: Other diseases of urinary system
        block_n30 = ICD10Node("N30-N39", "Other diseases of urinary system", 2)
        chapter_n.add_child(block_n30)
        
        # Category N39: Other disorders of urinary system
        cat_n39 = ICD10Node("N39", "Other disorders of urinary system", 3)
        block_n30.add_child(cat_n39)
        subcat_n39_0 = ICD10Node("N39.0", "Urinary tract infection, site not specified", 4)
        subcat_n39_1 = ICD10Node("N39.1", "Persistent proteinuria, unspecified", 4)
        subcat_n39_2 = ICD10Node("N39.2", "Orthostatic proteinuria, unspecified", 4)
        subcat_n39_3 = ICD10Node("N39.3", "Stress incontinence (female) (male)", 4)
        subcat_n39_4 = ICD10Node("N39.4", "Other specified urinary incontinence", 4)
        subcat_n39_8 = ICD10Node("N39.8", "Other specified disorders of urinary system", 4)
        subcat_n39_9 = ICD10Node("N39.9", "Disorder of urinary system, unspecified", 4)
        cat_n39.add_child(subcat_n39_0)
        cat_n39.add_child(subcat_n39_1)
        cat_n39.add_child(subcat_n39_2)
        cat_n39.add_child(subcat_n39_3)
        cat_n39.add_child(subcat_n39_4)
        cat_n39.add_child(subcat_n39_8)
        cat_n39.add_child(subcat_n39_9)
        
        # Build lookup maps
        self._build_maps(self.root)
    
    def _build_maps(self, node: ICD10Node):
        """Build lookup maps for quick access"""
        if node.code != "ROOT":
            self.code_map[node.code] = node
            
            # Build chapter and block maps
            if node.level == 1:  # Chapter
                self.chapter_map[node.code] = node.code
            elif node.level == 2:  # Block
                self.block_map[node.code] = node.code
        
        for child in node.children:
            self._build_maps(child)
    
    def _find_best_match(self, code: str) -> Optional[ICD10Node]:
        """Find best matching node for a given code"""
        # Direct match
        if code in self.code_map:
            return self.code_map[code]
        
        # Try to find parent category (e.g., I10 for I10.5)
        if '.' in code:
            base_code = code.split('.')[0]
            if base_code in self.code_map:
                return self.code_map[base_code]
        
        # Try to find by pattern matching
        for existing_code, node in self.code_map.items():
            if code.startswith(existing_code) or existing_code.startswith(code):
                return node
        
        # If no direct match, try to create a synthetic node based on ICD-10 structure
        # This handles cases where specific codes aren't in our hierarchy
        if len(code) >= 3:
            # Try to match at least to chapter level
            chapter_code = code[0]
            for existing_code, node in self.code_map.items():
                if existing_code.startswith(chapter_code) and '-' in existing_code:
                    # Found a chapter-level match, create a synthetic node
                    synthetic_node = ICD10Node(code, f"Code {code} (synthetic)", 3)
                    synthetic_node.parent = node
                    return synthetic_node
        
        return None
    
    def calculate_hierarchical_similarity(self, code1: str, code2: str) -> Dict[str, float]:
        """Calculate hierarchical similarity between two ICD-10 codes"""
        if code1 == code2:
            return {
                "exact_match": 1.0,
                "subcategory_match": 1.0,
                "category_match": 1.0,
                "block_match": 1.0,
                "chapter_match": 1.0,
                "hierarchical_distance": 0.0,
                "weighted_score": 1.0
            }
        
        node1 = self._find_best_match(code1)
        node2 = self._find_best_match(code2)
        
        if not node1 or not node2:
            return {
                "exact_match": 0.0,
                "subcategory_match": 0.0,
                "category_match": 0.0,
                "block_match": 0.0,
                "chapter_match": 0.0,
                "hierarchical_distance": 1.0,
                "weighted_score": 0.0
            }
        
        # Get paths to root
        path1 = node1.get_path_to_root()
        path2 = node2.get_path_to_root()
        
        # Find common ancestor level
        common_level = 0
        for i, (p1, p2) in enumerate(zip(path1, path2)):
            if p1 == p2:
                common_level = i
            else:
                break
        
        # Calculate matches at different levels
        chapter_match = len(path1) > 1 and len(path2) > 1 and path1[1] == path2[1]
        block_match = len(path1) > 2 and len(path2) > 2 and path1[2] == path2[2]
        category_match = len(path1) > 3 and len(path2) > 3 and path1[3] == path2[3]
        subcategory_match = len(path1) > 4 and len(path2) > 4 and path1[4] == path2[4]
        
        # Calculate hierarchical distance
        distance = (len(path1) - common_level - 1) + (len(path2) - common_level - 1)
        max_distance = 8  # Maximum possible distance in hierarchy
        normalized_distance = min(distance / max_distance, 1.0)
        
        # Calculate weighted score based on hierarchy level
        if category_match:
            weighted_score = 0.85
        elif block_match:
            weighted_score = 0.70
        elif chapter_match:
            weighted_score = 0.50
        else:
            weighted_score = max(0.0, 1.0 - normalized_distance)
        
        return {
            "exact_match": 0.0,
            "subcategory_match": 1.0 if subcategory_match else 0.0,
            "category_match": 1.0 if category_match else 0.0,
            "block_match": 1.0 if block_match else 0.0,
            "chapter_match": 1.0 if chapter_match else 0.0,
            "hierarchical_distance": normalized_distance,
            "weighted_score": weighted_score
        }
    
    def get_code_info(self, code: str) -> Dict[str, str]:
        """Get hierarchical information about a code"""
        node = self._find_best_match(code)
        if not node:
            return {"error": f"Code {code} not found in hierarchy"}
        
        path = node.get_path_to_root()
        info = {
            "code": code,
            "description": node.description,
            "level": node.level,
            "path": " -> ".join(path[1:])  # Skip ROOT
        }
        
        if len(path) > 1:
            info["chapter"] = path[1]
        if len(path) > 2:
            info["block"] = path[2]
        if len(path) > 3:
            info["category"] = path[3]
        if len(path) > 4:
            info["subcategory"] = path[4]
        
        return info

# Global instance
hierarchical_matcher = HierarchicalICD10Matcher()

def calculate_hierarchical_accuracy(predictions: List[Tuple[str, str]]) -> Dict[str, float]:
    """Calculate hierarchical accuracy for a list of (predicted, ground_truth) pairs"""
    if not predictions:
        return {}
    
    total_scores = {
        "exact_match": 0.0,
        "subcategory_match": 0.0,
        "category_match": 0.0,
        "block_match": 0.0,
        "chapter_match": 0.0,
        "weighted_score": 0.0
    }
    
    for predicted, ground_truth in predictions:
        similarity = hierarchical_matcher.calculate_hierarchical_similarity(predicted, ground_truth)
        for key in total_scores:
            total_scores[key] += similarity[key]
    
    # Calculate averages
    n = len(predictions)
    return {key: (value / n) * 100 for key, value in total_scores.items()}

def get_hierarchical_code_info(code: str) -> Dict[str, str]:
    """Get hierarchical information about an ICD-10 code"""
    return hierarchical_matcher.get_code_info(code)
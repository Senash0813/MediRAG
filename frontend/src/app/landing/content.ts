/** Static copy for the marketing landing page */

export const landingPipelines = [
  {
    name: 'Neurosciences',
    description:
      'A dedicated pipeline for questions spanning Neurology and Neurosurgery, covering disorders of the brain, spinal cord, and peripheral nervous system, as well as the surgical interventions that treat them. From epilepsy and stroke to spinal stenosis and brain tumors, this pipeline is built to handle the full diagnostic and therapeutic breadth of neuroscientific medicine.',
    ragFocus:
      'What sets this pipeline apart is its expert-mimicking scoring component. After retrieving candidate answers, it evaluates and ranks them the way a specialist would, weighing clinical relevance, anatomical precision, and evidence strength, before surfacing a final response.',
  },
  {
    name: 'Cardiovascular',
    description:
      'Covers the tightly interwoven fields of Cardiology, Cardiothoracic Surgery, and Vascular Surgery. These specialties share disease mechanisms like atherosclerosis, thrombosis, and aneurysms, along with common imaging modalities and overlapping clinical guidelines, making unified cross-specialty retrieval essential for accurate answers.',
    ragFocus:
      'This pipeline uses Hypothetical Document Embeddings (HyDE) combined with instruction-aware embeddings. Rather than matching your query directly, it first generates a hypothetical ideal answer and embeds that, dramatically improving retrieval precision for complex, cross-specialty cardiovascular questions.',
  },
  {
    name: 'Internal Medicine',
    description:
      'A broad-spectrum pipeline spanning General Internal Medicine, General Pediatrics, General Surgery, Nephrology, Endocrinology & Metabolism, Hematology, Pulmonology & Respiratory Medicine, and Gastroenterology & Hepatology, built to handle the diagnostic complexity that comes with overlapping, multi-system conditions.',
    ragFocus:
      'What distinguishes this pipeline is its multi-stage post-processing verification framework. Retrieved answers pass through successive validation layers, with each stage checking consistency, clinical plausibility, and evidence alignment, progressively narrowing down to the most fitting, well-supported response.',
  },
  {
    name: 'Primary Care & Mental Health',
    description:
      'Covers the human-centred end of medicine, including Psychiatry, Psychology & Behavioral Health, Nursing, Family Medicine & Primary Care, and Geriatrics. These specialties demand answers that are not only clinically accurate but sensitive to patient context, continuity of care, and holistic wellbeing.',
    ragFocus:
      'This pipeline combines two-phase evidence validation with a dynamic prompt strategy. The first phase retrieves and cross-checks candidate evidence, the second re-validates the shortlisted answers against clinical context, and the dynamic prompt adapts its instruction based on the question type, ensuring responses are appropriately nuanced across mental health, primary, and geriatric care scenarios.',
  },
] as const;

export const landingTestimonials = [
  {
    quote:
      'More reliable than general AI tools for clinical queries. The answers actually hold up when you fact-check them.',
    name: 'Dr. Mohamed Rikaz Sheriff',
    role: 'Medical Officer',
    org: 'Ministry of Health, Education & Research Unit',
  },
  {
    quote:
      'Noticeably less hallucination compared to ChatGPT on drug dosing questions. That alone makes it worth using.',
    name: 'Dr. Habeeba Sheriff',
    role: 'Senior House Officer (ICU)',
    org: 'Sri Jayewardanapura General Hospital',
  },
  {
    quote:
      'Better accuracy than I expected from an AI tool. Good starting point for looking up clinical guidelines.',
    name: 'Dr. Navanjana Warnakulasuriya',
    role: 'Registrar, Paediatric Surgery',
    org: 'Lady Ridgeway Hospital for Children',
  },
  {
    quote:
      'Really helpful when you need a quick, trustworthy reference during ward rounds. Doesn\'t make things up. Highly recommended for interns in teaching hospitals.',
    name: 'Dr. B.M. Madhini Basnayaka',
    role: 'Intern Medical Officer',
    org: 'Colombo South Teaching Hospital',
  },
] as const;

export const landingDevelopers = {
  title: 'Meet the team',
  paragraphs: [
    'MediRAG was developed by a team of final-year Data Science students at the Sri Lanka Institute of Information Technology (SLIIT) as part of their final year project.',
    'Driven by a shared passion for AI and its real-world impact, the team focused on solving a critical challenge in medical question answering, building a system that prioritizes accuracy, reliability, and trust over generic responses.',
    'Starting from the left, the team consists of Tharindu, Sandun, Senash, and Sajana, bringing together their skills and ideas to turn research into a practical, impactful solution.',
  ],
} as const;

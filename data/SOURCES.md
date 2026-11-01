# Corpus sources

Real public documents only, from official French sources. The machine-readable list used by
ingestion is `data/corpus/manifest.yaml`; only entries with `include: true` are indexed.

**Corpus as-of date: 2026-11-01** (extended from 2026-09-01). Documents published after that
date are out of scope.

| File | Title | Publisher | URL | Retrieved | Version shown in the document |
|---|---|---|---|---|---|
| `impots_faq_je-decouvre-la-fe.pdf` | Foire aux questions – Je découvre la facturation électronique (11 p.) | DGFiP | [link](https://www.impots.gouv.fr/sites/default/files/media/1_metier/2_professionnel/EV/2_gestion/290_facturation_electronique/faq---fe_je-decouvre-la-facturation-electronique.pdf) | 2026-11-01 | "Version du 01/09/2026" |
| `impots_fiche-1_que-va-t-il-se-passer.pdf` | Fiche 1 – Que va-t-il se passer pour mon entreprise en matière de facturation ? (3 p.) | DGFiP | [link](https://www.impots.gouv.fr/sites/default/files/media/1_metier/2_professionnel/EV/2_gestion/290_facturation_electronique/fiche-1_que-va-t-il-se-passer-pour-mon-entreprise.pdf) | 2026-11-01 | "Mise à jour juin 2026" |
| `impots_fiche-3_tpe_date.pdf` | Fiche 3 – À partir de quelle date mon entreprise doit-elle être prête… (2 p.) | DGFiP | [link](https://www.impots.gouv.fr/sites/default/files/media/1_metier/2_professionnel/EV/2_gestion/290_facturation_electronique/fiche-3_tpe_a-partir-de-quand-mon-entreprise-doit-etre-prete.pdf) | 2026-11-01 | "Mise à jour octobre 2026" |
| `impots_fiche-sci.pdf` | Facturation électronique : je suis une SCI (3 p.) | DGFiP | [link](https://www.impots.gouv.fr/sites/default/files/media/1_metier/2_professionnel/EV/2_gestion/290_facturation_electronique/fiches_reforme/fiche-sci.pdf) | 2026-11-01 | "Version octobre 2026" |
| `impots_plateformes-agreees.html` | Facturation électronique et plateformes agréées (web page) | DGFiP | [link](https://www.impots.gouv.fr/facturation-electronique-et-plateformes-agreees) | 2026-11-01 | No date on the page |
| `legifrance_cgi-ann2_242nonies_vigueur-2026-09-01.pdf` | CGI, annexe II, art. 242 nonies B à 242 nonies P (28 p.) | Légifrance | [link](https://www.legifrance.gouv.fr/codes/id/LEGISCTA000046385428/2026-09-01) | 2026-11-01 | "Version en vigueur au 01 septembre 2026" (consolidated) |
| `legifrance_ordonnance-2021-1190_version-initiale.pdf` | Ordonnance n° 2021-1190 du 15 septembre 2021 (10 p.) | Légifrance | [link](https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000044044176) | 2026-11-01 | **Version initiale** (JORF n°0216, 16/09/2021) – amended since |
| `legifrance_decret-2022-1299_version-initiale.pdf` | Décret n° 2022-1299 du 7 octobre 2022 (13 p.) | Légifrance | [link](https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000046383394) | 2026-11-01 | **Version initiale** (JORF n°0235, 09/10/2022) – amended since |

Notes:
- The three Légifrance files were saved by the repo owner from a browser (print to PDF),
  because Légifrance refuses automated downloads (HTTP 403). The page header/footer added by
  the browser print (date, URL, page number) is removed at ingestion.
- The ordonnance and the décret are the **original** Journal officiel versions. Several of
  their provisions, notably the 2024–2026 calendar, were later amended. The current rules are
  in the consolidated CGI annex and the DGFiP documents. The original texts are kept on
  purpose: their titles say "version initiale – dispositions modifiées depuis" so the agent
  can tell them apart, and the eval set can test whether the agent prefers current sources.
- Not in the corpus: CGI article 289 bis (consolidated). It can be added the same way.

## Synthetic material (not part of the corpus)

`data/adversarial/` holds synthetic documents written for the prompt-injection test cases.
They are not real sources, are listed in `data/adversarial/README.md`, and are indexed with
`collection: adversarial`, used only by `injection` eval cases.

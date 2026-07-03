% === Export pour PGFPlots (version corrigée) ===
% Entrées :
%   Coorneu : [Nbpt x 2]  coordonnées (x,y)
%   Numtri  : [Nbtri x 3] connectivité (indices MATLAB, base 1)
%   FONC      : [Nbpt x 1]  valeurs aux sommets
%
% Sorties :
%   sommets.dat   : x y u  (sans ligne d'en-tête)
%   triangles.dat : i j k (indices en base 0)

FONC = UU_eff;

% Vérifs
Nbpt  = size(Coorneu,1);
Nbtri = size(Numtri,1);
if length(FONC) ~= Nbpt
    error('La taille de FONC ne correspond pas au nombre de sommets');
end
if size(Coorneu,2) < 2
    error('Coorneu doit avoir au moins 2 colonnes (x,y).');
end

%% Export sommets (vectorisé)
fid = fopen('sommets_comp_eff_eps1.dat','w');
if fid < 0, error('Impossible d''ouvrir sommets.dat en écriture'); end
% IMPORTANT : on écrit uniquement des nombres (pas d'en-tête texte)
% fprintf lit la matrice colonne par colonne, donc on lui passe [x'; y'; u']
fprintf(fid, 'x y u\n'); % en-tête pour PGFPlots
fprintf(fid, '%g %g %g\n', [Coorneu(:,1)'; Coorneu(:,2)'; FONC']);
fclose(fid);

%% Export triangles (vectorisé, indices base 0)
fid = fopen('triangles_comp_eff_eps1.dat','w');
if fid < 0, error('Impossible d''ouvrir triangles.dat en écriture'); end
% Très important : transposer (Numtri-1) pour que fprintf écrive colonne par colonne,
% chaque colonne devenant une ligne dans le fichier.
fprintf(fid, '%d %d %d\n', (Numtri - 1)');   % <-- CORRECTION CLÉ
fclose(fid);

disp('Export terminé : sommets.dat et triangles.dat créés.');
